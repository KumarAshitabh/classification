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
curl -F 'image1=@sample_images/0.png' -F 'model=SVM' http://127.0.0.1:5000/predict

# If model name not specified 
 curl -F 'image1=@sample_images/0.png' -F 'model=""' http://127.0.0.1:5000/predict


#Some command history for reference
   40  cd classification/
   41  git status
   42  git checkout -b feature/final_exam
   43  python mymain.py --clf_name=Tree --randomseed=42
   44  python mymain.py --clf_name=Tree --randomseed=20
   45  ls model/
   46  cat report/svm.txt
   47  cat report/tree.txt
   48  curl -F 'image1=@sample_images/0.png'http://127.0.0.1:5000/predict
   49  curl -F 'image1=@sample_images/0.png' http://127.0.0.1:5000/predict
   50  curl -F 'image1=@sample_images/0.png' -F 'model=SVM' http://127.0.0.1:5000/predict
   51  curl -F 'image1=@sample_images/0.png' -F 'model=Tree' http://127.0.0.1:5000/predict
   52  sudo docker build --tag flask-docker-demo-app .
   53  sudo docker run --name flask-docker-demo-app -p 5000:5000 flask-docker-demo-app
   54  sudo docker rm flask-docker-demo-app
   55  sudo docker run --name flask-docker-demo-app -p 5000:5000 flask-docker-demo-app
   56  sudo docker rm flask-docker-demo-app
   57  sudo docker run --name flask-docker-demo-app -p 5000:5000 flask-docker-demo-app
   66  sudo docker rm flask-docker-demo-app
   67  sudo docker build --tag flask-docker-demo-app .
   68  sudo docker run --name flask-docker-demo-app -v /mnt/c/Ashitabh/Code/classification/model/:/tmp/  -p 5000:5000 flask-docker-demo-app

