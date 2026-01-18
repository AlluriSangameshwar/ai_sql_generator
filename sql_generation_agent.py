import pandas as pd
import os
from collections import defaultdict
from git import Repo, GitCommandError
import ollama
from pathlib import Path

# ---------------------------------------
# USER CONFIGURATION
# ---------------------------------------
CSV_FILE = "transformation_spec.csv"

GIT_REPO_URL = "https://github.com/AlluriSangameshwar/transformations_dbt.git"
LOCAL_REPO_PATH = "./transformations_dbt"
GIT_BRANCH = "main"

MODEL_NAME = "phi3:mini"  # ✅ lightweight local model

# ---------------------------------------
# STEP 1: READ INPUT METADATA (CSV)
# ---------------------------------------
def load_metadata(csv_file):
    try:
        return pd.read_csv(csv_file)
    except FileNotFoundError:
        raise RuntimeError(f"CSV file not found: {csv_file}")
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV: {e}")

# ---------------------------------------
# STEP 2: GROUP ROWS BY TARGET TABLE
# ---------------------------------------
def group_by_target_table(df):
    grouped = defaultdict(list)

    for _, row in df.iterrows():
        key = (row["tgt_dataset"], row["tgt_table"])
        grouped[key].append(row)

    return grouped

# ---------------------------------------
# STEP 3: BUILD PROMPT FOR LLM
# ---------------------------------------
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
- BigQuery SQL only
- SELECT statement only (no CREATE / INSERT)
- Proper column aliases
- If incremental load, use dbt is_incremental() logic
- Output ONLY SQL
"""
    return prompt.strip()

# ---------------------------------------
# STEP 4: CALL LOCAL LLM (PHI-MINI)
# ---------------------------------------
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

# ---------------------------------------
# STEP 5: WRITE SQL FILE
# ---------------------------------------
def write_sql(repo_path, dataset, table, sql):
    folder_path = os.path.join(repo_path, "models", dataset)
    os.makedirs(folder_path, exist_ok=True)

    file_path = os.path.join(folder_path, f"{table}.sql")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(sql.strip() + "\n")

    return file_path

# ---------------------------------------
# STEP 6: GIT COMMIT & PUSH
# ---------------------------------------
# def commit_and_push(repo_path, files):
#     try:
#         repo = Repo(repo_path)
#         repo.git.checkout(GIT_BRANCH)
#
#         for f in files:
#             repo.git.add(f)
#
#         repo.index.commit("Auto-generated BigQuery SQL via AI agent")
#         repo.remote(name="origin").push()
#
#     except GitCommandError as e:
#         raise RuntimeError(f"Git operation failed: {e}")


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
            print("✅ SQL committed & pushed to transformations_dbt")
        else:
            print("ℹ️ No changes detected")

    except GitCommandError as e:
        raise RuntimeError(f"Git operation failed: {e}")




# ---------------------------------------
# MAIN EXECUTION
# ---------------------------------------
def main():
    try:
        print("🚀 Starting SQL generation agent")

        df = load_metadata(CSV_FILE)
        grouped_tables = group_by_target_table(df)

        if not os.path.exists(LOCAL_REPO_PATH):
            print("📥 Cloning dbt repo")
            Repo.clone_from(GIT_REPO_URL, LOCAL_REPO_PATH)

        generated_files = []

        for target_key, rows in grouped_tables.items():
            print(f"⚙️ Generating SQL for {target_key[0]}.{target_key[1]}")
            prompt = build_prompt(target_key, rows)
            sql = generate_sql(prompt)

            file_path = write_sql(
                LOCAL_REPO_PATH,
                target_key[0],
                target_key[1],
                sql
            )
            generated_files.append(file_path)

        commit_and_push(LOCAL_REPO_PATH, generated_files)

        print("✅ BigQuery SQL generated and pushed successfully")

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")

if __name__ == "__main__":
    main()
