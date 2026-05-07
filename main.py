import warnings
warnings.filterwarnings("ignore", module="google")
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from database import engine, SessionLocal
from models import Base, Competition, Agent, CompetitionParticipant, Submission, Leaderboard, Rewards
import time
import threading
from datetime import datetime, timezone
import requests
import vertexai
from supabase import create_client
from vertexai.generative_models import GenerativeModel
from vertexai.preview.vision_models import ImageGenerationModel
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# create tables (safe)
Base.metadata.create_all(bind=engine)


class CreateCompetitionRequest(BaseModel):
    title: str
    prompt: str
    max_iterations: int
    min_agents: int
    duration: int = 60


class JoinCompetitionRequest(BaseModel):
    agent_id: int


class SubmitOutputRequest(BaseModel):
    competition_id: int
    agent_id: int
    image_url: str

#judge agent by GCC, changed from OpenRouter due to rate limits
from vertexai.generative_models import GenerativeModel, Part
import vertexai

def get_score_from_ai(prompt, image_url):
    try:
        vertexai.init(project="aicompetitionplatformagents", location="us-central1")

        model = GenerativeModel("gemini-2.5-flash")

        response = model.generate_content([
            f"Rate this image from 1 to 10.\n"

            f"Give a SHORT reason in 2–3 lines only:\n"
            f"- 1 line: what is good\n"
            f"- 1 line: what can be improved\n"

            f"Return format:\n"
            f"score: <number>, reason: <short text>\n",
            
            Part.from_uri(image_url, mime_type="image/png")
        ])

        text = response.text
        print("DEBUG:", text)

        score = float(text.split("score:")[1].split(",")[0].strip())
        reason = text.split("reason:", 1)[1].strip()

        return {"score": score, "reason": reason}

    except Exception as e:
        print("ERROR:", e)
        return {"score": 5.0, "reason": "error"}
        
    
# upload to supabase
url = "url"
key = "key_here"
supabase = create_client(url, key)

#mock_agent execution
def run_mock_agents(competition_id):
    db = SessionLocal()
    try:
        participants = db.query(CompetitionParticipant).filter(
            CompetitionParticipant.competition_id == competition_id
        ).all()
    finally:
        db.close()

    def start_agent_stream(agent_id):
        try:
            with requests.get(
                f"http://127.0.0.1:8000/stream-agent/{competition_id}/{agent_id}",
                stream=True,
                timeout=600,
            ) as response:
                for _ in response.iter_lines():
                    pass
        except Exception as e:
            print("STREAM START ERROR:", e)

    for participant in participants:
        threading.Thread(target=start_agent_stream, args=(participant.agent_id,), daemon=True).start()

#Add image generation

def generate_image_from_vertex(prompt):
    vertexai.init(project="aicompetitionplatformagents", location="us-central1")

    model = ImageGenerationModel.from_pretrained("imagen-4.0-fast-generate-001")
    images = model.generate_images(prompt=prompt)

    image_bytes = images[0]._image_bytes

    #file_name = f"temp_{int(time.time())}.png"
    file_name = f"{uuid.uuid4()}.png"
    supabase.storage.from_("images").upload(
        file_name,
        image_bytes,
        {"content-type": "image/png"}
    )

    public_url = supabase.storage.from_("images").get_public_url(file_name)

    return public_url


#added hugging face right now
def generate_image_from_huggingface(prompt):
    try:
        API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
        headers = {
            "Authorization": "Bearer key_here"
        }

        response = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": prompt}
        )

        if response.status_code != 200:
            print("HF ERROR:", response.text)
            return generate_image_from_vertex(prompt)

        image_bytes = response.content

        file_name = f"hf_{int(time.time())}.png"
        supabase.storage.from_("images").upload(
            file_name,
            image_bytes,
            {"content-type": "image/png"}
        )

        public_url = supabase.storage.from_("images").get_public_url(file_name)

        return public_url

    except Exception as e:
        print("ERROR:", e)
        return generate_image_from_vertex(prompt)


def generate_image(model_name, prompt):
    if model_name == "vertex":
        return generate_image_from_vertex(prompt)

    elif model_name == "huggingface":
        return generate_image_from_huggingface(prompt)

    else:
        return generate_image_from_vertex(prompt)

def call_gemini_stream(prompt):
    vertexai.init(project="aicompetitionplatformagents", location="us-central1")
    model = GenerativeModel("gemini-2.5-flash")
    for chunk in model.generate_content(prompt, stream=True):
        try:
            text = chunk.text
        except Exception:
            text = ""
        if text:
            yield text

def sse(text):
    return "data: " + text.replace("\n", "\ndata: ") + "\n\n"


def build_leaderboard_rows(submissions):
    best_scores = {}

    for submission in submissions:
        current_score = submission.score if submission.score is not None else 0
        best_entry = best_scores.get(submission.agent_id)
        best_score = best_entry["score"] if best_entry and best_entry["score"] is not None else 0

        if best_entry is None or current_score > best_score:
            best_scores[submission.agent_id] = {
                "agent_id": submission.agent_id,
                "score": submission.score,
                "reason": submission.reason,
            }

    ranked_rows = sorted(
        best_scores.values(),
        key=lambda row: row["score"] if row["score"] is not None else 0,
        reverse=True,
    )

    return [
        {
            **row,
            "rank": index + 1,
        }
        for index, row in enumerate(ranked_rows)
    ]


def persist_leaderboard(db, competition_id: int, leaderboard_rows):
    db.query(Leaderboard).filter(Leaderboard.competition_id == competition_id).delete(
        synchronize_session=False
    )

    for row in leaderboard_rows:
        db.add(
            Leaderboard(
                competition_id=competition_id,
                agent_id=row["agent_id"],
                final_score=row["score"],
                rank=row["rank"],
            )
        )


def finalize_competition(db, comp: Competition, submissions):
    leaderboard_rows = build_leaderboard_rows(submissions)
    persist_leaderboard(db, comp.id, leaderboard_rows)
    comp.status = "completed"
    db.commit()
    return leaderboard_rows


def split_generation_text(text):
    if not text or "FINAL_IMAGE_PROMPT:" not in text:
        return (text or "").strip(), ""

    reasoning_text, final_prompt = text.split("FINAL_IMAGE_PROMPT:", 1)
    return reasoning_text.strip(), final_prompt.strip()

@app.get("/stream-agent/{competition_id}/{agent_id}")
def stream_agent(competition_id: int, agent_id: int):
    db = SessionLocal()
    try:
        comp = db.query(Competition).filter(Competition.id == competition_id).first()
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not comp or not agent:
            raise HTTPException(status_code=404, detail="Not found")

        # ✅ FIRST: check existing submission
        existing = (
            db.query(Submission)
            .filter(
                Submission.competition_id == competition_id,
                Submission.agent_id == agent_id
            )
            .first()
        )

        if existing and existing.generation_status == "completed":
            db.close()

            def events_existing():
                if existing.reasoning_text:
                    yield sse(existing.reasoning_text)
                if existing.image_url:
                    yield sse(f"IMAGE_READY:{existing.image_url}")

            return StreamingResponse(events_existing(), media_type="text/event-stream")

        # ✅ THEN check competition status
        if comp.status != "ongoing":
            return StreamingResponse(iter([sse("Competition not active")]), media_type="text/event-stream")

        # ✅ ONLY THEN create new submission
        sub = (
            db.query(Submission)
            .filter(
                Submission.competition_id == competition_id,
                Submission.agent_id == agent_id,
                Submission.generation_status != "completed"
            )
            .first()
        )

        if not sub:
            sub = Submission(
                competition_id=competition_id,
                agent_id=agent_id,
                iteration_number=1,
                generation_status="streaming"
            )
            db.add(sub)
            db.commit()
            db.refresh(sub)

        competition_prompt = comp.prompt
        submission_id = sub.id
        model_name = agent.model_name

    finally:
        db.close()

    def events():
        db = SessionLocal()
        full_text = ""

        try:
            prompt = (
                f"Competition prompt: {competition_prompt}\n"
                " stream a brief first-person reasoning summary about what you will create like your chain of thought process. Before FINAL_IMAGE_PROMPT, write your own concise technical rationale for this specific image generation. "
                "Explain what you understood from the competition prompt, what visual decisions you are making, what constraints you are optimizing for, and why this prompt should produce a strong image."
    
                "then write FINAL_IMAGE_PROMPT: followed by the technical, and structured image prompt only."
               
            )

            # ✅ stream reasoning
            for chunk in call_gemini_stream(prompt):
                full_text += chunk

                sub = db.query(Submission).filter(Submission.id == submission_id).first()
                sub.reasoning_text = full_text
                db.commit()

                yield sse(chunk)

            # ✅ extract final prompt
            _, final_prompt = split_generation_text(full_text)
            image_prompt = final_prompt or competition_prompt

            # ✅ generate image
            image_url = generate_image(model_name, image_prompt)

            # ✅ save once
            sub = db.query(Submission).filter(Submission.id == submission_id).first()
            sub.reasoning_text = full_text
            sub.final_prompt = image_prompt
            sub.image_url = image_url
            sub.generation_status = "completed"
            db.commit()

            # ✅ send image
            yield sse(f"IMAGE_READY:{image_url}")

        except Exception as e:
            yield sse(f"ERROR: {e}")

        finally:
            db.close()

    return StreamingResponse(events(), media_type="text/event-stream")
#Agent apis
@app.post("/agent/register")
def register_agent(name: str, model_name: str="vertex"):
    db = SessionLocal()
    try:
        agent = Agent(name=name, model_name=model_name)
        db.add(agent)
        db.commit()
        return {"msg": "agent created", "agent_id": agent.id}
    finally:
        db.close()

@app.post("/agent/login")
def login_agent(id: int):
    db = SessionLocal()
    try:
        agent = db.query(Agent).filter(Agent.id == id).first()
        return {"msg": "login success" if agent else "not found"}
    finally:
        db.close()

#Competition apis

@app.post("/competition/create")
def create_competition(payload: CreateCompetitionRequest):
    db=SessionLocal()
    try:
        new_comp = Competition(
            title=payload.title,
            prompt=payload.prompt,
            max_iterations=payload.max_iterations,
            min_agents=payload.min_agents,
            duration=payload.duration
            
        )
        db.add(new_comp)
        db.commit()
        db.refresh(new_comp)
        return {
            "msg": "Competition created",
            "id": new_comp.id,
            "title": new_comp.title,
            "prompt": new_comp.prompt,
            "max_iterations": new_comp.max_iterations,
            "min_agents": new_comp.min_agents,
            "participant_count": 0,
            "status": new_comp.status,
            "created_at": new_comp.created_at,
        }
    finally:
        db.close()

@app.get("/competitions")
def get_competitions():
    db=SessionLocal()
    try:
        data=db.query(Competition).all()
        return [
            {
                "id": c.id,
                "title": c.title,
                "prompt": c.prompt,
                "max_iterations": c.max_iterations,
                "min_agents": c.min_agents,
                "participant_count": db.query(CompetitionParticipant).filter(CompetitionParticipant.competition_id == c.id).count(),
                "status": c.status,
                "created_at": c.created_at
            }
            for c in data
        ]
    finally:
        db.close()
@app.get("/competition/{id}")
def get_competition_by_id(id: int):
    db=SessionLocal()
    try:
        data=db.query(Competition).filter(Competition.id==id).first()
        if not data:
            raise HTTPException(status_code=404, detail="Competition not found")

        return {
            "id": data.id,
            "title": data.title,
            "prompt": data.prompt,
            "max_iterations": data.max_iterations,
            "min_agents": data.min_agents,
            "participant_count": db.query(CompetitionParticipant).filter(CompetitionParticipant.competition_id == data.id).count(),
            "status": data.status,
            "created_at": data.created_at
        }
    finally:
        db.close()

@app.get("/competition/{id}/participants")
def get_competition_participants(id: int):
    db = SessionLocal()
    try:
        competition = db.query(Competition).filter(Competition.id == id).first()
        if not competition:
            raise HTTPException(status_code=404, detail="Competition not found")

        participants = db.query(CompetitionParticipant).filter(
            CompetitionParticipant.competition_id == id
        ).order_by(CompetitionParticipant.id.asc()).all()
        agent_ids = [participant.agent_id for participant in participants]
        agents = db.query(Agent).filter(Agent.id.in_(agent_ids)).all() if agent_ids else []
        agent_map = {agent.id: agent for agent in agents}

        return [
            {
                "agent_id": participant.agent_id,
                "name": agent_map.get(participant.agent_id).name if agent_map.get(participant.agent_id) else f"Agent {participant.agent_id}",
                "model_name": agent_map.get(participant.agent_id).model_name if agent_map.get(participant.agent_id) else None,
            }
            for participant in participants
        ]
    finally:
        db.close()

@app.post("/competition/{id}/join")
def join_competition(id: int, payload: JoinCompetitionRequest):
    db = SessionLocal()
    try:
        competition = db.query(Competition).filter(Competition.id == id).first()
        if not competition:
            raise HTTPException(status_code=404, detail="Competition not found")
        if competition.status == "completed":
            raise HTTPException(status_code=400, detail="competition completed")

        existing_participant = db.query(CompetitionParticipant).filter(
            CompetitionParticipant.competition_id == id,
            CompetitionParticipant.agent_id == payload.agent_id
        ).first()
        if existing_participant:
            return {"msg": "already joined"}

        joined_count = db.query(CompetitionParticipant).filter(
            CompetitionParticipant.competition_id == id
        ).count()
        if joined_count >= competition.min_agents:
            raise HTTPException(status_code=400, detail="competition full")

        agent = db.query(Agent).filter(Agent.id == payload.agent_id).first()
        if not agent:
            agent = Agent(id=payload.agent_id, name=f"Agent {payload.agent_id}")
            db.add(agent)
            db.commit()

        cp = CompetitionParticipant(competition_id=id, agent_id=payload.agent_id)
        db.add(cp)
        db.commit()
        return {"msg": "joined"}
    finally:
        db.close()

#Submission apis

@app.post("/submit-output")
def submit_output(payload: SubmitOutputRequest):
    db = SessionLocal()
    try:
        sub = Submission(
            competition_id=payload.competition_id,
            agent_id=payload.agent_id,
            image_url=payload.image_url
        )
        db.add(sub)
        db.commit()
        return {"msg": "submitted"}
    finally:
        db.close()

@app.get("/competition/{id}/submissions")
def get_submissions(id: int):
    db = SessionLocal()
    try:
        data = db.query(Submission).filter(Submission.competition_id == id).all()
        return [
            {
                "agent_id": s.agent_id,
                "submission_id": s.id,
                "image_url": s.image_url,
                "score": s.score,
                "reason": s.reason,
                "reasoning_text": s.reasoning_text,
                "final_prompt": s.final_prompt,
                "generation_status": s.generation_status,
                "iteration_number": s.iteration_number,
            }
            for s in data
        ]
    finally:
        db.close()

#leaderboard apis
@app.get("/leaderboard/{competition_id}")
def leaderboard(competition_id: int):
    db = SessionLocal()
    try:
        stored_rows = (
            db.query(Leaderboard)
            .filter(Leaderboard.competition_id == competition_id)
            .order_by(Leaderboard.rank.asc(), Leaderboard.final_score.desc())
            .all()
        )
        if stored_rows:
            return [
                {
                    "agent_id": row.agent_id,
                    "score": row.final_score,
                    "reason": None,
                    "rank": row.rank,
                }
                for row in stored_rows
            ]

        submissions = db.query(Submission).filter(Submission.competition_id == competition_id).all()
        return build_leaderboard_rows(submissions)
    finally:
        db.close()

#Internal Apis(mock for now)
@app.post("/internal/start-competition/{id}")
def start_comp(id: int):
    db = SessionLocal()
    try:
        comp = db.query(Competition).filter(Competition.id == id).first()

        if not comp:
            return {"msg": "not found"}
        if comp.status != "upcoming":
            raise HTTPException(status_code=400, detail="competition already started or completed")

        comp.status = "ongoing"
        db.commit()

        run_mock_agents(id)

        def run_later(comp_id, duration):
            time.sleep(duration)
            evaluate(comp_id)

        threading.Thread(target=run_later, args=(id, comp.duration)).start()

        return {"msg": "competition started"}
    finally:
        db.close()



@app.post("/internal/run-iteration")
def run_iter():
    return {"msg": "iteration run"}




@app.post("/internal/evaluate/{competition_id}")
def evaluate(competition_id: int):
    db = SessionLocal()
    try:
        comp = db.query(Competition).filter(Competition.id == competition_id).first()
        if not comp:
            raise HTTPException(status_code=404, detail="competition not found")

        if comp.status == "upcoming":
            raise HTTPException(status_code=400, detail="competition not started")

        subs = db.query(Submission).filter(Submission.competition_id == competition_id).all()

        for submission in subs:
            if submission.score is not None:
                    continue  # ✅ already scored, skip
            result = get_score_from_ai(comp.prompt, submission.image_url)
            submission.score = result["score"]
            submission.reason = result["reason"]

        participants = db.query(CompetitionParticipant).filter(
            CompetitionParticipant.competition_id == competition_id
        ).all()

        can_complete = len(participants) >= (comp.min_agents or 0)
        required_iterations = comp.max_iterations or 1

        for participant in participants:
            scored_count = 0
            for submission in subs:
                if submission.agent_id == participant.agent_id and submission.score is not None:
                    scored_count += 1

            if scored_count < required_iterations:
                can_complete = False
                break

        if can_complete:
            finalize_competition(db, comp, subs)
            return {"msg": "AI scoring done and competition completed"}

        db.commit()
        return {"msg": "AI scoring done. Waiting for all agents."}
    finally:
        db.close()

#Rewards:
@app.post("/internal/distribute-rewards")
def rewards():
    return {"msg": "rewards distributed"}



@app.post("/internal/update-leaderboard/{competition_id}")
def update_leaderboard(competition_id: int):
    db = SessionLocal()
    try:
        comp = db.query(Competition).filter(Competition.id == competition_id).first()
        if not comp:
            return {"msg": "not found"}
        if comp.status == "upcoming":
            raise HTTPException(status_code=400, detail="competition not started")

        subs = db.query(Submission).filter(Submission.competition_id == competition_id).all()
        participants = db.query(CompetitionParticipant).filter(
            CompetitionParticipant.competition_id == competition_id
        ).all()

        can_complete = len(participants) >= (comp.min_agents or 0)
        required_iterations = comp.max_iterations or 1

        for participant in participants:
            scored_count = 0
            for submission in subs:
                if submission.agent_id == participant.agent_id and submission.score is not None:
                    scored_count += 1

            if scored_count < required_iterations:
                can_complete = False
                break

        if not can_complete:
            raise HTTPException(status_code=400, detail="waiting for all agents to finish scoring")

        finalize_competition(db, comp, subs)
        return {"msg": "leaderboard rebuilt"}
    finally:
        db.close()




#health api

@app.get("/health")
def health():
    return {"status": "ok"}
