from urllib.parse import quote_plus
from pymongo import MongoClient

# MongoDB connection string
user=quote_plus("Deva@0813")

client=MongoClient("mongodb+srv://Devanand:"+user+"@devadb.mi8y86o.mongodb.net/?retryWrites=true&w=majority")

db=client["login_info"]
col=db["login0"]

#Validate email and password
def validate(email,password): 
    if email and password:
        data=data=col.find_one({"email":email,"password":password})
        if data:
            return True
        else:
            return False
    else:
        return False
    
#Insert data into database
def insert(name,email,password,disability,role): 
    if name and check(email) and password and disability and role:
        query={"name":name,"email":email,"password":password,"disability":disability,"role":role}
        col.insert_one(query)
        return True
    else:
        return False
    
#Update data in database
def update(name,email,age):
    if not check(email):
        if name and email and age:
            col.update_many({"email":email}, {"$set":{"name":name}})
            return True
        else:
            return False
    else:
        return False
    
#Check if email already exists
def check(email): 
    if email:
        data=col.find_one({"email":email})
        if data:
            return False
        else:
            return True
        
#show all data in database
def show(email): 
    if email:
        data=col.find_one({"email":email},{"_id":0})
        return data
    else:
        return False
