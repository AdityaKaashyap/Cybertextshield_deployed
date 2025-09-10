from fastapi import FastAPI, HTTPException
from bson import ObjectId
from database import task_collection
from models import Task, TaskUpdate
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

# Allow frontend (React) to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Helper to convert MongoDB documents
def task_helper(task) -> dict:
    return {
        "id": str(task["_id"]),
        "title": task["title"],
        "completed": task["completed"],
        "created_at": task["created_at"]
    }

@app.get("/tasks")
async def get_tasks():
    tasks = []
    async for task in task_collection.find():
        tasks.append(task_helper(task))
    return tasks

@app.post("/tasks")
async def create_task(task: Task):
    new_task = task.dict()
    result = await task_collection.insert_one(new_task)
    created_task = await task_collection.find_one({"_id": result.inserted_id})
    return task_helper(created_task)

@app.put("/tasks/{task_id}")
async def update_task(task_id: str, task: TaskUpdate):
    update_data = {k: v for k, v in task.dict().items() if v is not None}
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")
    result = await task_collection.update_one({"_id": ObjectId(task_id)}, {"$set": update_data})
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Task not found")
    updated_task = await task_collection.find_one({"_id": ObjectId(task_id)})
    return task_helper(updated_task)

@app.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    result = await task_collection.delete_one({"_id": ObjectId(task_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"message": "Task deleted"}
