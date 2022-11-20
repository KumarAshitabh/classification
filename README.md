# mlops-22
Final Exam

# Build and Run docker image for our Flask App
sudo docker build --tag flask-docker-demo-app .
sudo docker run --name flask-docker-demo-app -p 5000:5000 flask-docker-demo-app

# Check running containers
docker ps -a


# To remove container for any code changes)
sudo docker rm flask-docker-demo-app

# Validate API using curl
curl -F 'image1=@sample_images/0.png' -F 'image2=@sample_images/5.png'  http://127.0.0.1:5000/predict