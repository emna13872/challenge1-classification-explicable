# Image unique : construit l'interface React puis la sert depuis FastAPI (compatible Hugging Face Spaces, port 7860).
FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PORT=7860
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /app
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
# Jeu de Taïwan (UCI n°350, licence CC BY 4.0) téléchargé à la construction : sans lui, CreditLens n'a pas son jeu
# principal en ligne. En cas d'échec, l'image se construit quand même (l'application dira que le fichier manque).
# Le jeu de fraude (150 Mo, Kaggle) n'est pas inclus : il reste disponible seulement en local.
RUN mkdir -p /app/data && python -c "import io, urllib.request, zipfile; \
z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen('https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip', timeout=120).read())); \
z.extractall('/app/data'); print('Taiwan :', z.namelist())" || echo "Jeu de Taïwan non téléchargé"
COPY backend/ ./backend/
COPY --from=web /web/dist ./frontend/dist
RUN chown -R user /app/data
USER user
WORKDIR /app/backend
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
