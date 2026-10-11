import json
import os
from dotenv import load_dotenv
from groq import Groq
from paddle import Model

load_dotenv()

client = Groq()

def load_json_file(file_path):
    if not os.path.exists(file_path):
        print(f"file {file_path} was not found!")
        return ""
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data.get("raw_text", "")


def request_response(cv_text, job_text):
    prompt = f"""
        Analyze the candidate CV text and Job Description text provided below.
        Extract only essential key qualifications, experience gaps, and background facts.
        Return a tightly structured, concise summary for a voice agent context window.

        [Candidate CV Raw Text]
        {cv_text}

        [Job Description Raw Text]
        {job_text}

        Output Requirements:
        Return strictly a valid JSON object with the following keys:
        1. "candidate": name, total_years_experience, top_5_skills, latest_role, key_projects
        2. "job": target_role, required_skills, min_years_experience
        3. "gap_analysis": matching_skills, missing_skills, suggested_interview_questions
        Ensure the JSON is well-formed and parsable. Do not include any extra text or commentary."""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    # alternative model: 
        # meta-llama/llama-prompt-guard-2-86m
        # canopylabs/orpheus-v1-english
        # whisper-large-v3-turbo
        # openai/gpt-oss-safeguard-20b
        # allam-2-7b
        # whisper-large-v3
        # canopylabs/orpheus-arabic-saudi
        # openai/gpt-oss-20b
        # qwen/qwen3.8-27b
        # openai/gpt-oss-120b
        # meta-llama/llama-prompt-guard-2-22m


    return json.loads(response.choices[0].message.content)


if __name__ == "__main__":

    cv_data = load_json_file("./data/cv_data.json")
    job_data = load_json_file("./data/job_data.json")

    result = request_response(cv_data, job_data)

    output_file = "./data/summarized CV & JD.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            result,
            f, 
            ensure_ascii=False,
            indent=4)

    print("Done 3")
