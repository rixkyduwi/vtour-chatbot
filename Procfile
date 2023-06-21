web: gunicorn wsgi:app
heroku ps:scale web=1
gunicorn --worker-tmp-dir /dev/shm vtour-chatbot.wsgi