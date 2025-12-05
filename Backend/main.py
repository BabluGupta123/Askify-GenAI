# from fastapi import FastAPI, UploadFile, Form
# from fastapi.middleware.cors import CORSMiddleware
# from loader import load_pdf
# from rag import create_vector_db, ask_question
# import os

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Global DB object
# vector_db = None


# @app.post("/upload-pdf/")
# async def upload_pdf(file: UploadFile):
#     global vector_db

#     pdf_path = f"uploaded/{file.filename}"
#     os.makedirs("uploaded", exist_ok=True)

#     with open(pdf_path, "wb") as f:
#         f.write(await file.read())

#     docs = load_pdf(pdf_path)
#     vector_db = create_vector_db(docs)

#     return {"status": "PDF processed successfully!", "pages": len(docs)}


# @app.post("/ask/")
# async def ask(query: str = Form(...)):
#     global vector_db
#     if not vector_db:
#         return {"error": "Please upload a PDF first"}

#     answer = ask_question(query, vector_db)
#     return {"answer": answer}



from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from loader import load_pdf
from rag import create_vector_db, ask_question
import os
from qdrant_client import QdrantClient

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global DB object
vector_db = None


@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile):
    global vector_db

    pdf_path = f"uploaded/{file.filename}"
    os.makedirs("uploaded", exist_ok=True)

    # Save uploaded PDF
    with open(pdf_path, "wb") as f:
        f.write(await file.read())

    # Load PDF and split into chunks
    docs = load_pdf(pdf_path)

    # -------------------------------
    # 🔥 IMPORTANT: CLEAR OLD PDF DATA
    # -------------------------------
    client = QdrantClient(url="http://localhost:6333")

    try:
        client.delete_collection(collection_name="pdf_rag")
        print("Old collection deleted!")
    except Exception as e:
        print("No existing collection to delete:", e)

    # -------------------------------
    # Create fresh vector DB for new PDF
    # -------------------------------
    vector_db = create_vector_db(docs)

    return {"status": "PDF processed successfully!", "pages": len(docs)}


@app.post("/ask/")
async def ask(query: str = Form(...)):
    global vector_db

    if not vector_db:
        return {"error": "Please upload a PDF first"}

    answer = ask_question(query, vector_db)
    return {"answer": answer}
