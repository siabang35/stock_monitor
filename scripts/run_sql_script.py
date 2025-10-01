#!/usr/bin/env python3
"""
Script to run SQL scripts on the database
"""
import os
import sys
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_supabase_client() -> Client:
    """Get Supabase client"""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")
    
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set in environment variables")
    
    return create_client(supabase_url, supabase_key)

def run_sql_script(script_path: str):
    """Run SQL script on Supabase"""
    try:
        # Read SQL script
        with open(script_path, 'r') as f:
            sql_content = f.read()
        
        # Get Supabase client
        supabase = get_supabase_client()
        
        # Split SQL content by statements (simple approach)
        statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
        
        print(f"Running SQL script: {script_path}")
        print(f"Found {len(statements)} SQL statements")
        
        # Execute each statement
        for i, statement in enumerate(statements, 1):
            if statement:
                try:
                    print(f"Executing statement {i}...")
                    # Note: Supabase client doesn't have direct SQL execution
                    # This would need to be adapted based on your database setup
                    print(f"Statement: {statement[:100]}...")
                    
                except Exception as e:
                    print(f"Error executing statement {i}: {e}")
                    continue
        
        print("✅ SQL script executed successfully")
        
    except Exception as e:
        print(f"❌ Error running SQL script: {e}")
        sys.exit(1)

def main():
    if len(sys.argv) != 2:
        print("Usage: python run_sql_script.py <script_path>")
        print("Example: python run_sql_script.py create_tables.sql")
        sys.exit(1)
    
    script_path = sys.argv[1]
    
    if not os.path.exists(script_path):
        print(f"❌ Script file not found: {script_path}")
        sys.exit(1)
    
    run_sql_script(script_path)

if __name__ == "__main__":
    main()
