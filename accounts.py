import sqlite3
import hashlib
import json
import os

class AccountManager:
    def __init__(self, db_path="accounts.db"):
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        """Initializes the database and creates the accounts table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS accounts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    name TEXT NOT NULL,
                    mastery_level TEXT DEFAULT 'Beginner',
                    progress TEXT DEFAULT '{}'
                )
            ''')
            conn.commit()
            
    def _hash_password(self, password):
        """Hashes the password using SHA-256."""
        return hashlib.sha256(password.encode()).hexdigest()
        
    def create_account(self, username, password, name, mastery_level='Beginner'):
        """Creates a new account."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Initialize progress for A-Z with level 1
                initial_progress = {chr(i): 1 for i in range(ord('A'), ord('Z')+1)}
                
                cursor.execute('''
                    INSERT INTO accounts (username, password_hash, name, mastery_level, progress)
                    VALUES (?, ?, ?, ?, ?)
                ''', (username, self._hash_password(password), name, mastery_level, json.dumps(initial_progress)))
                conn.commit()
                return True, "Account created successfully."
        except sqlite3.IntegrityError:
            return False, "Username already exists."
            
    def login(self, username, password):
        """Logs in a user by checking the username and password."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, name, mastery_level, progress FROM accounts WHERE username = ? AND password_hash = ?', 
                           (username, self._hash_password(password)))
            result = cursor.fetchone()
            if result:
                return True, {
                    "id": result[0],
                    "username": username,
                    "name": result[1],
                    "mastery_level": result[2],
                    "progress": json.loads(result[3])
                }
            return False, "Invalid username or password."
            
    def get_account(self, username):
        """Retrieves account details without needing a password."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, name, mastery_level, progress FROM accounts WHERE username = ?', (username,))
            result = cursor.fetchone()
            if result:
                return {
                    "id": result[0],
                    "username": username,
                    "name": result[1],
                    "mastery_level": result[2],
                    "progress": json.loads(result[3])
                }
            return None
            
    def update_account(self, username, name=None, mastery_level=None):
        """Updates general account details."""
        updates = []
        params = []
        if name:
            updates.append("name = ?")
            params.append(name)
        if mastery_level:
            updates.append("mastery_level = ?")
            params.append(mastery_level)
            
        if not updates:
            return False, "No updates provided."
            
        params.append(username)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                UPDATE accounts 
                SET {", ".join(updates)}
                WHERE username = ?
            ''', params)
            conn.commit()
            return cursor.rowcount > 0, "Account updated."
            
    def _calculate_mastery_level(self, progress: dict) -> str:
        """Calculates mastery level based on total progress across all letters."""
        total_score = sum(progress.values())
        if total_score < 73:
            return "Beginner"
        elif total_score < 120:
            return "Intermediate"
        elif total_score < 167:
            return "Advanced"
        elif total_score < 214:
            return "Expert"
        else:
            return "Mastery"

    def update_progress(self, username, letter, level):
        """Updates the progress level (1-10) for a specific sign language letter."""
        if not (1 <= level <= 10):
            return False, "Progress level must be between 1 and 10."
            
        letter = letter.upper()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Fetch current progress in one query
            cursor.execute('SELECT progress FROM accounts WHERE username = ?', (username,))
            result = cursor.fetchone()
            
            if not result:
                return False, "Account not found."
            
            progress = json.loads(result[0])
            progress[letter] = level
            
            new_mastery_level = self._calculate_mastery_level(progress)
            
            # Update in the same connection
            cursor.execute('UPDATE accounts SET progress = ?, mastery_level = ? WHERE username = ?', 
                           (json.dumps(progress), new_mastery_level, username))
            conn.commit()
            return True, f"Progress for '{letter}' updated to {level}. Mastery is now {new_mastery_level}."

    def change_password(self, username: str, old_password: str, new_password: str):
        """Change an account's password after verifying the old password."""
        if not new_password:
            return False, "New password must not be empty."

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Verify existing password
            cursor.execute('SELECT password_hash FROM accounts WHERE username = ?', (username,))
            row = cursor.fetchone()
            if not row:
                return False, "Account not found."

            current_hash = row[0]
            if self._hash_password(old_password) != current_hash:
                return False, "Old password is incorrect."

            # Update to new password hash
            new_hash = self._hash_password(new_password)
            cursor.execute('UPDATE accounts SET password_hash = ? WHERE username = ?', (new_hash, username))
            conn.commit()
            return True, "Password changed successfully."
            
    def delete_account(self, username):
        """Deletes an account."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM accounts WHERE username = ?', (username,))
            conn.commit()
            return cursor.rowcount > 0, "Account deleted."

# Example Usage (You can remove this when importing into other files)
if __name__ == "__main__":
    manager = AccountManager()
    
    # 1. Create account
    success, msg = manager.create_account("johndoe", "secret123", "John Doe")
    print("Create:", msg)
    
    # 2. Login
    success, data = manager.login("johndoe", "secret123")
    print("Login:", "Success" if success else "Failed", data)
    
    # 3. Update Progress for 'A' to level 5
    success, msg = manager.update_progress("johndoe", "A", 5)
    print("Update Progress:", msg)
    
    # 4. View updated account
    print("Updated Account:", manager.get_account("johndoe"))
