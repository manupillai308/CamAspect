import pyrebase
import glob

config = {
	"apiKey": "AIzaSyAxm000LOCAVudGYDQsdPtqR2OHFfMHYZU",
	"authDomain": "fir-search-e9674.firebaseapp.com",
	"databaseURL": "https://fir-search-e9674.firebaseio.com",
	"projectId": "fir-search-e9674",
	"storageBucket": "fir-search-e9674.appspot.com",
	"messagingSenderId": "45962151411"
  }


firebase = pyrebase.initialize_app(config)


auth = firebase.auth()

email="admin@gmail.com"
password = "123qwerty"


user = auth.sign_in_with_email_and_password(email, password)

db = firebase.database()
storage = firebase.storage()

def put_data(name, first_seen, last_seen):		
	all_users = db.child("Users").get()
	person = {}
	for user in all_users.each():
		if user.val().get("nothing") is not None:
			break
		person[user.val()['name']] = user.key()
	person_name = glob.glob('./images/%s*' % name.lower())[0]
	person_id = person.get(name)
	storage.child("profile_images/%s" % person_name.split('/')[-1]).put(person_name)
	url = storage.child("profile_images/%s" % person_name.split('/')[-1]).get_url(token=None)
	if person_id is None:
		person_id = len(person) + 1
	else:
		person_id = int(person_id)
	data = {"Users/0%d"% person_id:{"image":url, "first":first_seen, "name":name.lower(), "status":last_seen}}
	db.update(data)
