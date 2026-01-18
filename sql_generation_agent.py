import pandas as pd
import os
from collections import defaultdict
from git import Repo, GitCommandError
import ollama
import tempfile
from pathlib import Path

CSV_FILE = "transformation_spec.csv"

GIT_REPO_URL = "https://github.com/AlluriSangameshwar/transformations_dbt.git"
GIT_BRANCH = "main"

MODEL_NAME = "phi3:mini"

TEMP_BASE_DIR = tempfile.gettempdir()
MODEL_NAME = "phi3:mini"
CSV_FILE = "transformation_spec.csv"

def load_metadata(csv_file):
    try:
        return pd.read_csv(csv_file)
    except FileNotFoundError:
        raise RuntimeError(f"CSV file not found: {csv_file}")
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV: {e}")


def group_by_target_table(df):
    grouped = defaultdict(list)

    for _, row in df.iterrows():
        key = (row["tgt_dataset"], row["tgt_table"])
        grouped[key].append(row)

    return grouped


def build_prompt(target_key, rows):
    tgt_dataset, tgt_table = target_key

    src_project = rows[0]["src_project"]
    src_dataset = rows[0]["src_dataset"]
    src_table = rows[0]["src_table"]

    column_instructions = [
        f"- {r['src_column']} → {r['tgt_column']} : {r['transformation_rule']}"
        for r in rows
    ]

    filter_condition = rows[0].get("filter_condition", "")
    load_type = rows[0].get("load_type", "full")
    watermark = rows[0].get("watermark_column", "")

    prompt = f"""
You are a BigQuery SQL expert.

Generate a BigQuery SELECT query using the rules below.
**Do not use Markdown, code fences, or any extra formatting.**
**Do not include CREATE or INSERT statements.**
**Use TIMESTAMP_ADD for time arithmetic, IS_NAN for numeric checks, and proper backticks for table references.**

SOURCE:
{src_project}.{src_dataset}.{src_table}

TARGET:
{tgt_dataset}.{tgt_table}

COLUMN RULES:
{chr(10).join(column_instructions)}

FILTER:
{filter_condition}

LOAD TYPE:
{load_type}

WATERMARK COLUMN:
{watermark}

REQUIREMENTS:
- BigQuery Standard SQL only
- SELECT statement only
- Proper column aliases
- Incremental load: use dbt is_incremental() logic
- Output ONLY SQL, no Markdown or triple backticks
"""
    return prompt.strip()

def generate_sql(prompt):
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0,
                "num_ctx": 2048
            }
        )
        print(response["message"]["content"].strip())
        return response["message"]["content"].strip()

    except Exception as e:
        raise RuntimeError(f"Ollama SQL generation failed: {e}")


def write_sql(repo_path, table, sql):
    output_dir = os.path.join(repo_path, "generated_sql")
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, f"{table}.sql")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(sql.strip() + "\n")

    return file_path


from pathlib import Path
from git import Repo, GitCommandError

def commit_and_push(repo_path, files):
    try:
        repo = Repo(repo_path)

        # 🚿 Force clean state
        repo.git.fetch("origin")
        repo.git.checkout(GIT_BRANCH)
        repo.git.reset("--hard", f"origin/{GIT_BRANCH}")

        repo_root = Path(repo_path).resolve()

        for f in files:
            rel_path = Path(f).resolve().relative_to(repo_root)
            repo.git.add(str(rel_path))

        if repo.is_dirty():
            repo.index.commit("AI: auto-generate BigQuery SQL")
            repo.remote("origin").push()
            print("SQL committed & pushed to transformations_dbt")
        else:
            print("No changes detected")

    except GitCommandError as e:
        raise RuntimeError(f"Git operation failed: {e}")


def main():
    try:
        print("Starting AI SQL generator")

        df = load_metadata(CSV_FILE)
        grouped_tables = group_by_target_table(df)

        # 🧊 Clone repo into TEMP directory
        temp_repo_path = os.path.join(
            TEMP_BASE_DIR,
            "ai_sql_push_repo"
        )

        if os.path.exists(temp_repo_path):
            Repo(temp_repo_path).git.reset("--hard")
        else:
            Repo.clone_from(GIT_REPO_URL, temp_repo_path, branch=GIT_BRANCH)

        generated_files = []

        for (tgt_dataset, tgt_table), rows in grouped_tables.items():
            print(f"⚙️ Generating SQL for {tgt_table}")

            prompt = build_prompt((tgt_dataset, tgt_table), rows)
            sql = generate_sql(prompt)

            file_path = write_sql(
                temp_repo_path,
                tgt_table,
                sql
            )
            generated_files.append(file_path)

        commit_and_push(temp_repo_path, generated_files)

        print("SQL generated & pushed (no local artifacts created)")

    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    main()