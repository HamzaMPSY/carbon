from flask import Flask

app = Flask(__name__)

# Import the routes module to register routes
from app import routes

