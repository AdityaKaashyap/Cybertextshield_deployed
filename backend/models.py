from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class Task(BaseModel):
    title:str
    completed:bool = False
    created_at:datetime = datetime.utcnow()

class TaskUpdate(BaseModel):
    titlr:Optional[str] = None
    completed:Optional[bool] = None