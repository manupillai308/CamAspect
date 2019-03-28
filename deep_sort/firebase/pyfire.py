import pyrebase
import glob

# CODE OMMITTED

user = auth.sign_in_with_email_and_password(email, password)

db = firebase.database()
storage = firebase.storage()

def put_data(name, first_seen, last_seen):		
	all_users = db.child("Users").get()
	users = {}
	person = {}
	for user in all_users.each():
		if user.val().get("nothing") is not None:
			break
		users[user.key()] = user.val()
		person[user.val()['name']] = user.key()
	person_name = glob.glob('./images/%s*' % name.lower())[0]
	person_id = person.get(name.lower())
	storage.child("profile_images/%s" % person_name.split('/')[-1]).put(person_name)
	url = storage.child("profile_images/%s" % person_name.split('/')[-1]).get_url(token=None)
	tracks = {}
	if person_id is None:
		person_id = len(person) + 1
		tracks["0%d" % 1] = {"first_seen":first_seen, "last_seen":last_seen, "image": url }
	else:
		person_id = int(person_id)
		tracks = users["0%d" % person_id]["tracks"]
		track_no = len(tracks) + 1
		tracks["0%d" % track_no] = {"first_seen":first_seen, "last_seen":last_seen, "image": url }
	data = {"Users/0%d"% person_id:{"image":url, "first":first_seen, "name":name.lower(), "status":last_seen, "tracks":tracks}}
	db.update(data)
