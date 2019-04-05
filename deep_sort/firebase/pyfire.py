import pyrebase
import glob

# CODE OMMITTED

user = auth.sign_in_with_email_and_password(email, password)

db = firebase.database()
storage = firebase.storage()

def put_data(name, first_seen, last_seen, person_image):		
	all_users = db.child("Users").get()
	users = {}
	person = {}
	for user in all_users.each():
		if user.val().get("nothing") is not None:
			break
		users[user.key()] = user.val()
		person[user.val()['name']] = user.key()
	person_name = glob.glob('./images/%s*' % name.lower())[0]
	person_image_name = "./log_images/%s.jpg" % (str(time.time()).split('.')[0])
	cv2.imwrite(person_image_name, person_image)
	person_id = person.get(name)
	storage.child("profile_images/%s" % person_name.split('/')[-1]).put(person_name)
	storage.child("profile_images/%s"  % person_image_name.split('/')[-1]).put(person_image_name)
	url = storage.child("profile_images/%s" % person_name.split('/')[-1]).get_url(token=None)
	person_url = storage.child("profile_images/%s" % person_image_name.split('/')[-1]).get_url(token=None)
	os.remove(person_image_name)
	tracks = {}
	if person_id is None:
		person_id = len(person) + 1
		tracks["0%d" % 1] = {"first_seen":first_seen, "last_seen":last_seen, "image": person_url }
	else:
		person_id = int(person_id)
		tracks = users["0%d" % person_id]["tracks"]
		track_no = len(tracks) + 1
		tracks["0%d" % track_no] = {"first_seen":first_seen, "last_seen":last_seen, "image": person_url, "id":"0%d" % person_id }
	data = {"Users/0%d"% person_id:{"image":url, "first":first_seen, "name":name, "status":last_seen, "tracks":tracks, "id":"0%d" % person_id}}
	db.update(data)
