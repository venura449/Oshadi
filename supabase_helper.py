import os
from dotenv import load_dotenv
from datetime import datetime
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL=os.getenv("SUPABASE_URL")
SUPABASE_KEY=os.getenv("SUPABASE_KEY")
STORAGE_BUCKET=os.getenv("STORAGE_BUCKET")

supabase: Client =create_client(SUPABASE_URL,SUPABASE_KEY)

def upload_alert(frame_path, camera_name, alert_type, message, room_count):
        try:
                file_name = os.path.basename(frame_path)

                with open(frame_path, 'rb') as f:
                        storage_path = f"snapshots/{file_name}"
                        supabase.storage.from_(STORAGE_BUCKET).upload(storage_path, f)

                snapshot_url = supabase.storage.from_(STORAGE_BUCKET).get_public_url(storage_path)

                data = {
                        "camera":camera_name,
                        "alert_type": alert_type,
                        "message": f"{message} (Cound:{room_count})",
                        "snapshot_url":snapshot_url,
                        "created_at": datetime.now().isoformat()
                }

                response = supabase.table("alerts").insert(data).execute()
                print(f"Alert pushed to database: {alert_type}")
                return response
        except Exception as e:
                print(f"supabase error {e}")
                return None
