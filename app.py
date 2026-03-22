import os
import re
import json
import uuid
import traceback
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
You are a professional content strategist. Given the transcript below, generate platform-optimized content using EXACTLY the section headers and item separators shown. Output plain text only — no JSON, no markdown, no code fences, no extra commentary.

=== LINKEDIN ===
--- POST 1 ---
[write LinkedIn post 1 here]
--- POST 2 ---
[write LinkedIn post 2 here]
--- POST 3 ---
[write LinkedIn post 3 here]

=== TWITTER ===
--- THREAD 1 ---
[TWEET] [write tweet 1 here]
[TWEET] [write tweet 2 here]
[TWEET] [write tweet 3 here]
--- THREAD 2 ---
[TWEET] [write tweet 1 here]
[TWEET] [write tweet 2 here]
[TWEET] [write tweet 3 here]
--- THREAD 3 ---
[TWEET] [write tweet 1 here]
[TWEET] [write tweet 2 here]
--- THREAD 4 ---
[TWEET] [write tweet 1 here]
[TWEET] [write tweet 2 here]
--- THREAD 5 ---
[TWEET] [write tweet 1 here]
[TWEET] [write tweet 2 here]

=== NEWSLETTER ===
SUBJECT: [subject line here]
[newsletter body here]

=== INSTAGRAM ===
--- CAPTION 1 ---
[write caption 1 here]
--- CAPTION 2 ---
[write caption 2 here]
--- CAPTION 3 ---
[write caption 3 here]

=== BLOG ===
TITLE: [blog title here]
[blog summary here]

Rules:
- linkedin: 3 distinct posts, professional tone, hook in first line, short paragraphs, 3-5 hashtags each.
- twitter: 5 threads of 3-6 tweets each. Each tweet <= 280 chars, numbered (1/, 2/, ...), emojis, last tweet is a CTA.
- newsletter: compelling subject line on the SUBJECT: line, then a structured body with intro, key points, takeaway.
- instagram: conversational, emojis throughout, 10-15 hashtags at the very end.
- blog: title on the TITLE: line, then a 150-250 word summary suitable as a meta-description.

TRANSCRIPT:
"""


def parse_sections(raw):
    """Parse Claude's delimiter-separated response into a structured dict."""

    def extract_section(text, header):
        m = re.search(
            rf"===\s*{header}\s*===\s*(.*?)(?=\n===\s*[A-Z]+\s*===|\Z)",
            text, re.DOTALL | re.IGNORECASE,
        )
        return m.group(1).strip() if m else ""

    def split_named_items(text, prefix):
        parts = re.split(rf"---\s*{prefix}\s*\d+\s*---", text, flags=re.IGNORECASE)
        return [p.strip() for p in parts if p.strip()]

    # LinkedIn — 3 posts
    li_raw = extract_section(raw, "LINKEDIN")
    linkedin = [{"post": p} for p in split_named_items(li_raw, "POST")] or [{"post": li_raw}]

    # Twitter — 5 threads, each containing [TWEET] markers
    tw_raw = extract_section(raw, "TWITTER")
    threads_raw = split_named_items(tw_raw, "THREAD")
    twitter = []
    for thread_text in threads_raw:
        tweets = [t.strip() for t in re.split(r"\[TWEET\]", thread_text) if t.strip()]
        if tweets:
            twitter.append({"thread": tweets})

    # Newsletter — SUBJECT: line + body
    nl_raw = extract_section(raw, "NEWSLETTER")
    nl_match = re.match(r"SUBJECT:\s*(.+?)[\r\n]+(.*)", nl_raw, re.DOTALL | re.IGNORECASE)
    if nl_match:
        newsletter = {"subject": nl_match.group(1).strip(), "section": nl_match.group(2).strip()}
    else:
        newsletter = {"subject": "", "section": nl_raw}

    # Instagram — 3 captions
    ig_raw = extract_section(raw, "INSTAGRAM")
    instagram = [{"caption": c} for c in split_named_items(ig_raw, "CAPTION")] or [{"caption": ig_raw}]

    # Blog — TITLE: line + summary
    bl_raw = extract_section(raw, "BLOG")
    bl_match = re.match(r"TITLE:\s*(.+?)[\r\n]+(.*)", bl_raw, re.DOTALL | re.IGNORECASE)
    if bl_match:
        blog = {"title": bl_match.group(1).strip(), "summary": bl_match.group(2).strip()}
    else:
        blog = {"title": "", "summary": bl_raw}

    return {
        "linkedin":   linkedin,
        "twitter":    twitter,
        "newsletter": newsletter,
        "instagram":  instagram,
        "blog":       blog,
    }


@app.route("/generate", methods=["POST"])
def generate():
    raw = ""
    try:
        data = request.get_json(silent=True) or {}
        transcript = data.get("transcript", "").strip()

        if not transcript:
            return jsonify({"error": "No transcript provided."}), 400

        # Use streaming so gunicorn stays active during the long API call
        chunks = []
        with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": GENERATE_PROMPT + transcript,
                }
            ],
        ) as stream:
            for text in stream.text_stream:
                chunks.append(text)

        raw = "".join(chunks).strip()
        print("[generate] raw response length:", len(raw))

        result = parse_sections(raw)
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "raw": raw}), 500


if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    app.run(debug=True)
