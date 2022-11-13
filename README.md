# mlops-22
Take home quiz


sudo docker build --tag flask-docker-demo-app .
sudo docker run --name flask-docker-demo-app -p 5000:5000 flask-docker-demo-app

# Check running containers
docker ps -a


#To remove container for any code changes)
sudo docker rm flask-docker-demo-app
