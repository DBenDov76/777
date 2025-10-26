📘 הוראות להעלאת האתר 777 ל-Render.com

בתיקייה הזו יש את כל מה שאתה צריך:
- app.py — קובץ Flask הראשי (מותאם ל-Render)
- templates/index.html — הדף שלך
- requirements.txt — הספריות הנדרשות
- Procfile — מורה ל-Render איך להריץ
- run_app.bat — לשימוש מקומי בלבד (לא נחוץ ב-Render)
- outputs/ — תיקייה ריקה לשמירת קבצים זמניים

🪜 שלבים פשוטים:

1️⃣ כנס ל-[https://github.com](https://github.com)
   - לחץ על "New Repository"
   - תן שם: 777
   - לחץ "Create Repository"

2️⃣ העלה את כל הקבצים שבתיקייה הזו (כולל התיקייה templates)
   - לחץ "Upload files"
   - גרור פנימה את כל התוכן מתוך התיקייה Flask_777_Render_20251026_1314
   - לחץ "Commit changes"

3️⃣ כנס ל-[https://render.com](https://render.com)
   - לחץ "New +" → "Web Service"
   - חבר את החשבון שלך ל-GitHub
   - בחר את הריפו בשם 777

4️⃣ בשדות:
   - Build Command:
     pip install -r requirements.txt
   - Start Command:
     python app.py
   - בחר תוכנית: Free
   - לחץ "Create Web Service"

5️⃣ חכה 2–3 דקות לסיום ההעלאה.
   כשתראה "Build successful 🎉" — האתר שלך באוויר!
   הקישור יהיה משהו כמו:
   https://777.onrender.com

6️⃣ טיפים:
   - כל עדכון ב-GitHub יגרום ל-Deploy חדש אוטומטית.
   - אם האתר "נרדם", פשוט כנס שוב — Render יעיר אותו.

בהצלחה 🎯
