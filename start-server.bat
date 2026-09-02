@echo off
cd /d "%~dp0"
set EMAIL_SENDER=akshay080408@gmail.com
set EMAIL_APP_PASSWORD=myshzxbuvudfhphk
set ADMIN_RECIPIENTS=rhar88802@gmail.com
set CITIZEN_RECIPIENTS=bemaxx768@gmail.com,kalayarra815@gmail.com,kumaraswami108008@gmail.com,dentovareceptionist@gmail.com,gaddamgopikrishna7@gmail.com
set EMAIL_TEST_MODE=false
python api/app.py
pause