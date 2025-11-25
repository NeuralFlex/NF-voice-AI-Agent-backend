from fastapi import APIRouter, Form, File, UploadFile, HTTPException
from pydantic import BaseModel
import uuid
from typing import Optional

from utils.cv_tools import extract_text_from_pdf_bytes, chunk_cv_text
from services.vectorstore_service import create_vectorstore, delete_vectorstore
from graph.graph import compiled_graph

router = APIRouter(tags=["Interview"])


class ContinueRequest(BaseModel):
    """Request model for continuing an existing interview session."""

    user_response: str
    thread_id: str


@router.post("/start_interview")
async def start_interview(
    job_title: str = Form(...),
    question_type: str = Form(...),
    cv: Optional[UploadFile] = File(None),
):
    """
    Start a new interview session.

    Args:
        job_title (str): The title or role for which the interview is being conducted.
        question_type (str): The type of questions (e.g., technical, behavioral).
        cv (Optional[UploadFile]): Optional CV file to extract contextual data from.

    Returns:
        dict: The initial interview question and session details.

    Raises:
        HTTPException: If interview initialization fails.
    """
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    user_id = "user123"
    cv_text = ""

    try:
        if cv:
            cv_bytes = await cv.read()
            cv_text = extract_text_from_pdf_bytes(cv_bytes)
            document = chunk_cv_text(cv_text, user_id=user_id)
            create_vectorstore(document, user_id=user_id)

        initial_state = {
            "topic": job_title,
            "content": [],
            "cv_content": cv_text[:1000] if cv_text else "",
            "questions": [],
            "answers": [],
            "user_response": None,
            "feedback": [],
            "current_question": None,
            "current_answer": None,
            "step": 0,
            "max_steps": 4,
            "final_evaluation": None,
            "messages": [],
            "question_type": question_type,
            "needs_retrieval": False,
            "retrieved_context": None,
            "similarity_score": None,
            "user_id": user_id,
        }

        final_state = compiled_graph.invoke(initial_state, config=config)

        return {
            "thread_id": thread_id,
            "status": "question",
            "message": final_state["messages"][-1]["content"],
            "current_step": final_state["step"],
            "max_steps": final_state["max_steps"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start interview: {e}")


@router.post("/continue_interview")
async def continue_interview(req: ContinueRequest):
    """
    Continue an existing interview session by providing a user's response.

    Args:
        req (ContinueRequest): The user's response and associated thread ID.

    Returns:
        dict: The next interview question or final evaluation.

    Raises:
        HTTPException: If continuation or state retrieval fails.
    """
    config = {"configurable": {"thread_id": req.thread_id}}

    try:
        existing_state = compiled_graph.get_state(config)
        if not existing_state:
            raise HTTPException(
                status_code=400, detail="No ongoing interview for this thread."
            )

        if hasattr(existing_state, "values"):
            state_dict = dict(existing_state.values)
        else:
            state_dict = dict(existing_state)

        state_dict["user_response"] = req.user_response
        state_dict["waiting_for_user"] = False

        final_state = compiled_graph.invoke(state_dict, config=config)
        messages = final_state.get("messages", [])

        if messages and messages[-1].get("role") == "system":
            return {
                "thread_id": req.thread_id,
                "status": "completed",
                "message": messages[-1]["content"],
                "current_step": final_state.get("step", 1),
                "max_steps": final_state.get("max_steps", 3),
            }

        if final_state.get("feedback"):
            user_id = final_state.get("user_id", "default_user")
            delete_vectorstore(user_id)

            return {
                "thread_id": req.thread_id,
                "status": "completed",
                "message": final_state["feedback"],
                "current_step": final_state.get("step", 1),
                "feedback_list": final_state["feedback"],
                "final_evaluation": final_state["final_evaluation"],
                "max_steps": final_state.get("max_steps", 3),
            }

        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                return {
                    "thread_id": req.thread_id,
                    "status": "question",
                    "message": msg["content"],
                    "current_step": final_state.get("step", 1),
                    "max_steps": final_state.get("max_steps", 3),
                }

        if messages:
            return {
                "thread_id": req.thread_id,
                "status": "unknown",
                "message": messages[-1]["content"],
                "current_step": final_state.get("step", 1),
                "max_steps": final_state.get("max_steps", 3),
            }

        raise HTTPException(
            status_code=500, detail="No response generated from the graph."
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to continue interview: {e}"
        )
