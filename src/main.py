from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .database import Base, engine
from .routes import user_routes, analysis_routes
import warnings
warnings.filterwarnings("ignore", message=".*bcrypt version.*")


Base.metadata.create_all(bind=engine)

app = FastAPI(title="FastAPI Auth Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://raml.elhlwgy.com",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(user_routes.router, prefix="/api")
app.include_router(analysis_routes.router, prefix="/api")

@app.get("/")
def root():
    return {"message": "FastAPI Auth Backend Running"}
