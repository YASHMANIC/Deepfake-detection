from app import app, User, db
import os

def view_users():
    try:
        db_path = app.config['SQLALCHEMY_DATABASE_URI']
        print(f"Database URI: {db_path}")
        
        with app.app_context():
            # Check if the table exists
            tables = db.engine.table_names()
            print(f"Available tables: {tables}")
            
            users = User.query.all()
            if not users:
                print("No users found in the database.")
            else:
                print("\nRegistered Users:")
                print("-" * 50)
                for user in users:
                    print(f"Username: {user.username}")
                    print(f"Email: {user.email}")
                    print("-" * 50)
    except Exception as e:
        print(f"Error: {str(e)}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Database file exists: {os.path.exists('instance/app.db')}")

if __name__ == "__main__":
    view_users()
