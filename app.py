import os
import re
import json
import uuid
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import anthropic
from openai import OpenAI

load_dotenv()

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.environ.get("UPLOAD_FOLDER", "/tmp/content-repurposer-uploads")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB limit for audio/video

ALLOWED_EXTENSIONS = {"txt", "md", "pdf"}
ALLOWED_AUDIO_EXTENSIONS = {"mp4", "mp3", "wav", "m4a"}

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

REPURPOSE_FORMATS = {
    "twitter_thread": "Convert the following content into an engaging Twitter/X thread. Use numbered tweets (1/, 2/, etc.), keep each tweet under 280 characters, add relevant emojis, and end with a call to action.",
    "linkedin_post": "Transform the following content into a professional LinkedIn post. Make it insightful, add a hook in the first line, use short paragraphs, include 3-5 relevant hashtags at the end.",
    "blog_post": "Rewrite the following content as a well-structured blog post. Include an engaging title, introduction, subheadings (using ##), body paragraphs, and a conclusion with a call to action.",
    "email_newsletter": "Convert the following content into an email newsletter. Include a subject line suggestion, a friendly greeting, organized sections with clear headings, key takeaways, and a sign-off.",
    "youtube_script": "Transform the following content into a YouTube video script. Include a hook for the first 15 seconds, clear sections, spoken-word friendly language, on-screen text suggestions in [brackets], and an outro with subscribe CTA.",
    "instagram_caption": "Convert the following content into an Instagram caption. Start with an attention-grabbing first line, make it conversational, add line breaks for readability, include relevant emojis, and end with 10-15 hashtags.",
}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html", formats=list(REPURPOSE_FORMATS.keys()))


@app.route("/repurpose", methods=["POST"])
def repurpose():
    content = ""

    # Handle file upload
    if "file" in request.files and request.files["file"].filename:
        file = request.files["file"]
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed. Use .txt or .md files."}), 400

        filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        finally:
            os.remove(filepath)  # Clean up temp file

    # Handle pasted text
    elif request.form.get("text_content"):
        content = request.form.get("text_content", "").strip()

    if not content:
        return jsonify({"error": "No content provided. Paste text or upload a file."}), 400

    target_format = request.form.get("format")
    if target_format not in REPURPOSE_FORMATS:
        return jsonify({"error": "Invalid format selected."}), 400

    prompt = REPURPOSE_FORMATS[target_format]

    try:
        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": f"{prompt}\n\n---\n\n{content}",
                }
            ],
        )
        result = message.content[0].text
        return jsonify({"result": result, "format": target_format})

    except anthropic.APIError as e:
        return jsonify({"error": f"API error: {str(e)}"}), 500


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files or not request.files["file"].filename:
        return jsonify({"error": "No file provided."}), 400

    file = request.files["file"]
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""

    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type '.{ext}'. Allowed: mp4, mp3, wav, m4a."}), 400

    filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        with open(filepath, "rb") as audio_file:
            response = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )
        return jsonify({"transcript": response.text})

    except Exception as e:
        return jsonify({"error": f"Transcription failed: {str(e)}"}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


GENERATE_PROMPT = """\
You are a professional content strategist. Given the transcript below, generate platform-optimized content for every format listed. You MUST return ONLY a valid JSON object — no markdown, no code fences, no explanation — with exactly these five keys:

{
  "linkedin": [
    {"post": "..."},
    {"post": "..."},
    {"post": "..."}
  ],
  "twitter": [
    {"thread": ["tweet 1", "tweet 2", "..."]},
    {"thread": ["tweet 1", "tweet 2", "..."]},
    {"thread": ["tweet 1", "tweet 2", "..."]},
    {"thread": ["tweet 1", "tweet 2", "..."]},
    {"thread": ["tweet 1", "tweet 2", "..."]}
  ],
  "newsletter": {
    "subject": "...",
    "section": "..."
  },
  "instagram": [
    {"caption": "..."},
    {"caption": "..."},
    {"caption": "..."}
  ],
  "blog": {
    "title": "...",
    "summary": "..."
  }
}

Rules per format:
- linkedin: 3 distinct posts, professional tone, hook in first line, short paragraphs, 3–5 hashtags each.
- twitter: 5 separate threads, each thread is an array of tweets (3–7 tweets), each tweet ≤ 280 chars, numbered (1/, 2/, …), relevant emojis, ends with a CTA.
- newsletter: 1 newsletter section with a compelling subject line and well-structured body (intro, key points, takeaway).
- instagram: 3 captions, conversational, emojis, 10–15 hashtags at the end of each.
- blog: title + 150–250 word summary suitable as an intro or meta-description.

TRANSCRIPT:
"""


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or {}
    transcript = data.get("transcript", "").strip()

    if not transcript:
        return jsonify({"error": "No transcript provided."}), 400

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": GENERATE_PROMPT + transcript,
                }
            ],
        )
        raw = message.content[0].text.strip()

        # Strip accidental markdown code fences if Claude adds them
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        result = json.loads(raw)

        required_keys = {"linkedin", "twitter", "newsletter", "instagram", "blog"}
        missing = required_keys - result.keys()
        if missing:
            return jsonify({"error": f"Model response missing keys: {missing}", "raw": raw}), 500

        return jsonify(result)

    except json.JSONDecodeError as e:
        return jsonify({"error": f"Model returned invalid JSON: {str(e)}", "raw": raw}), 500
    except anthropic.APIError as e:
        return jsonify({"error": f"Anthropic API error: {str(e)}"}), 500


if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    app.run(debug=True)
