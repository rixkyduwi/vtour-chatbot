from application import app
import os
if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 5000)))
    #app.run(host="0.0.0.0",port=4040,debug=True)