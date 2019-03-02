import datetime
from deep_sort.firebase.pyfire import *
from multiprocessing import Process

class TrackState:

	Tentative = 1
	Confirmed = 2
	Deleted = 3


class Track:
	
	def __init__(self, mean, covariance, track_id, n_init, max_age,
				 feature=None):
		self.mean = mean
		self.covariance = covariance
		self.track_id = track_id
		self.hits = 1
		self.age = 1
		self.time_since_update = 0

		self.state = TrackState.Tentative
		self.features = []
		if feature is not None:
			self.features.append(feature)

		self._n_init = n_init
		self._max_age = max_age
		
		self.face_image = None
		self.name = None
		self.first_seen = datetime.datetime.strftime(datetime.datetime.now(), "%H:%M [%d-%m-%y]")
		self.last_seen = None
		

	def to_tlwh(self):
		ret = self.mean[:4].copy()
		ret[2] *= ret[3]
		ret[:2] -= ret[2:] / 2
		return ret

	def to_tlbr(self):
		ret = self.to_tlwh()
		ret[2:] = ret[:2] + ret[2:]
		return ret

	def predict(self, kf):
		self.mean, self.covariance = kf.predict(self.mean, self.covariance)
		self.age += 1
		self.time_since_update += 1

	def update(self, kf, detection):
		self.mean, self.covariance = kf.update(
			self.mean, self.covariance, detection.to_xyah())
		self.features.append(detection.feature)
		self.hits += 1
		self.time_since_update = 0
		if self.state == TrackState.Tentative and self.hits >= self._n_init:
			self.state = TrackState.Confirmed

	def mark_missed(self):
		if self.state == TrackState.Tentative:
			self.state = TrackState.Deleted
		elif self.time_since_update > self._max_age:
			self.last_seen = datetime.datetime.strftime(datetime.datetime.now(), "%H:%M [%d-%m-%y]")
			self.state = TrackState.Deleted
			if self.name is not None:
				p = Process(target=put_data, args=(self.name, self.first_seen, self.last_seen))
				p.start()
			

	def is_tentative(self):
		return self.state == TrackState.Tentative

	def is_confirmed(self):
		return self.state == TrackState.Confirmed

	def is_deleted(self):
		return self.state == TrackState.Deleted
