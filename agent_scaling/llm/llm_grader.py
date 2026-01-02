import os
import json
import re

def _strip_json_code_block(text: str) -> str:
    """
    Remove markdown code block formatting (```json ... ```) from LLM output.
    """
    text = text.strip()
    # Remove triple backtick code block with optional 'json' language
    code_block_pattern = r"^```(?:json)?\s*([\s\S]*?)\s*```$"
    match = re.match(code_block_pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text

def llm_grade(model_output: str, reference: str, task_type: str = "qa", model_name: str = "gpt-4o") -> dict:
    """
    Use an LLM to grade the model output against the reference answer.
    Supports OpenAI (gpt-4o, gpt-4, etc.) and Gemini (model_name starting with 'gemini').
    Args:
        model_output: The model's answer to grade.
        reference: The reference/ground truth answer.
        task_type: Task type for prompt context.
        model_name: LLM model to use for grading (e.g., 'gpt-4o', 'gemini-pro').
    Returns:
        dict with keys: correct (bool), score (float), explanation (str)
    """
    prompt = f"""
    You are an expert evaluator. Given the following model answer and reference answer, decide if the model answer is correct. 
    Model answer: {model_output}
    Reference answer: {reference}
    Respond with JSON: {{"correct": true/false, "score": float (0-1), "explanation": "..."}}
    """
    if model_name.startswith("gemini"):
        import google.generativeai as genai  # type: ignore
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set.")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        # Gemini's response is in response.text
        text = response.text or ""
        if not text:
            print("LLM grader returned empty response!")
            return {"correct": 0, "score": 0.0, "explanation": "LLM grader returned empty response."}
        try:
            cleaned = _strip_json_code_block(text)
            return json.loads(cleaned)
        except Exception as e:
            print("Failed to parse LLM grader response as JSON:", text)
            return {"correct": 0, "score": 0.0, "explanation": f"Failed to parse LLM grader response: {text}"}
    else:
        import openai
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = response.choices[0].message.content
        if not content:
            print("LLM grader returned empty response!")
            return {"correct": 0, "score": 0.0, "explanation": "LLM grader returned empty response."}
        try:
            cleaned = _strip_json_code_block(content)
            return json.loads(cleaned)
        except Exception as e:
            print("Failed to parse LLM grader response as JSON:", content)
            return {"correct": 0, "score": 0.0, "explanation": f"Failed to parse LLM grader response: {content}"}