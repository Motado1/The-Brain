The Brain 🧠

Mission
Centralized AI-augmented command centre for projects, knowledge, tasks and learning.

Quick Start

# clone & env
git clone https://github.com/<you>/the-brain.git .
cp .env.example .env

# spin up Supabase
cd backend/supabase
supabase start

# in a new terminal, start everything else
cd ~/the-brain
docker compose up -d

Tech Stack

Supabase (Postgres)

Qdrant (vector DB)

n8n (workflows)

FastAPI (AI-Router)

Next.js (frontend)

