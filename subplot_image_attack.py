import numpy as np
import cv2
import matplotlib.pyplot as plt

# Fixing random state for reproducibility
np.random.seed(19680801)
# plot_contents = ['ptx','pose']
plot_contents = ['ptx']
for content in plot_contents:
	if content == 'pose':
		n_file_per_person = 3
	if content=='ptx':
		n_file_per_person = 6

	n_person_per_row =2
	n_row = 4
	n_col = n_file_per_person * n_person_per_row
	fig, axs = plt.subplots(nrows=n_row, ncols=n_col, figsize=(20, 20),
							subplot_kw={'xticks': [], 'yticks': []})
	plt.tight_layout()

	subplot_id= 1
	n_person = 12
	prefix= 'plot/save'
	ptx_files = list()
	depth_files = list()
	pncc_files= list()
	pose_files =list()
	ptx_files2 = list()
	depth_files2 = list()
	pncc_files2 = list()
	pose_files2 = list()

	ptx_files3 = list()
	depth_files3 = list()
	pncc_files3 = list()
	pose_files3 = list()
	# person_id_list = [1,2, 3]
	for person_id in range(1, n_person+1):
		ptx_file_name = prefix + str(person_id) +'orig_gt.jpg'
		ptx_files.append(cv2.imread(ptx_file_name, cv2.IMREAD_COLOR))
		ptx_file_name2 = prefix + str(person_id) +'attack_cnn.jpg'
		ptx_files2.append(cv2.imread(ptx_file_name2, cv2.IMREAD_COLOR))

		depth_file_name = prefix + str(person_id)+ '_depth' +'attack_cnn.png'
		depth_files.append(cv2.imread(depth_file_name, cv2.IMREAD_COLOR))
		depth_file_name2 = prefix + str(person_id)+ '_depth' +'attack_gbdt.png'
		depth_files2.append(cv2.imread(depth_file_name2, cv2.IMREAD_COLOR))
		
		pncc_file_name = prefix + str(person_id)+ '_pncc' +'attack_cnn.png'
		pncc_files.append(cv2.imread(pncc_file_name, cv2.IMREAD_COLOR))
		pncc_file_name2 = prefix + str(person_id)+ '_pncc' +'attack_gbdt.png'
		pncc_files2.append(cv2.imread(pncc_file_name2, cv2.IMREAD_UNCHANGED))

		pose_file_name = prefix + str(person_id)+ '_pose' +'orig_gt.jpg'
		pose_files.append(cv2.imread(pose_file_name, cv2.IMREAD_UNCHANGED))
		pose_file_name2 = prefix + str(person_id)+ '_pose' +'attack_cnn.jpg'
		pose_files2.append(cv2.imread(pose_file_name2, cv2.IMREAD_UNCHANGED))
		pose_file_name3 = prefix + str(person_id)+ '_pose' +'attack_gbdt.jpg'
		pose_files3.append(cv2.imread(pose_file_name3, cv2.IMREAD_UNCHANGED))

	subplot_id = 1
	person_id = 0
	if content=='ptx':
		for ax in axs.flat:
			# print(ax[0])
			# note = ' '
			if subplot_id%n_file_per_person==1:
				ax.imshow(ptx_files[person_id][:,:,::-1])
				if person_id<n_person_per_row: ax.set_title('GT Label')
			if subplot_id%n_file_per_person==2:
				ax.imshow(ptx_files2[person_id][:,:,::-1])
				if person_id<n_person_per_row: ax.set_title('CNN Attacked')
			if subplot_id%n_file_per_person==3:
				ax.imshow(depth_files[person_id])
				if person_id<n_person_per_row: ax.set_title('CNN Attacked')
			if subplot_id%n_file_per_person==0:
				ax.imshow(depth_files2[person_id])
				if person_id<n_person_per_row: ax.set_title('GBDT Attacked')
			if subplot_id%n_file_per_person==5:
				ax.imshow(pncc_files[person_id])
				if person_id<3: ax.set_title('CNN Attacked')
			if subplot_id%n_file_per_person==4:
				ax.imshow(pncc_files2[person_id])
				if person_id<3: ax.set_title('GBDT Attacked')

			subplot_id += 1
			if subplot_id%n_file_per_person==0:
				person_id += 1
	subplot_id = 0
	person_id = 0
	if content =='pose':
		for ax in axs.flat:
			# print(ax[0])
			# note = ' '
			if subplot_id%n_file_per_person==0:
				ax.imshow(pose_files[person_id][:,:,::-1])
				if person_id<n_person_per_row: ax.set_title('GT Label')
			if subplot_id%n_file_per_person==1:
				ax.imshow(pose_files2[person_id][:,:,::-1])
				if person_id<n_person_per_row: ax.set_title('CNN Attacked')
			if subplot_id%n_file_per_person==2:
				ax.imshow(pose_files3[person_id][:,:,::-1])
				if person_id<n_person_per_row: ax.set_title('GBDT Attacked')
			# print("person_id:"+str(person_id))
			subplot_id += 1
			if subplot_id%n_file_per_person==0:
				person_id += 1
	savefile = 'plot/comparison_' + content + '.png'
	plt.savefig(savefile, dpi=100)
	plt.show()
