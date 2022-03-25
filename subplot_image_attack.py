import numpy as np
import cv2
import matplotlib.pyplot as plt

# Fixing random state for reproducibility
np.random.seed(19680801)

grid = np.random.rand(4, 4)

n_file_per_person = 4 
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
ptx_files2 = list()
depth_files2 = list()
pncc_files2 = list()
# person_id_list = [1,2, 3]
for person_id in range(1, n_person+1):
	ptx_file_name = prefix + str(person_id) +'orig_gt.jpg'
	ptx_files.append(cv2.imread(ptx_file_name, cv2.IMREAD_COLOR))
	ptx_file_name2 = prefix + str(person_id) +'attack_cnn.jpg'
	ptx_files2.append(cv2.imread(ptx_file_name2, cv2.IMREAD_COLOR))

	depth_file_name = prefix + str(person_id)+ '_depth' +'attack_cnn.png'
	depth_files.append(cv2.imread(depth_file_name, cv2.IMREAD_UNCHANGED))
	depth_file_name2 = prefix + str(person_id)+ '_depth' +'attack_gbdt.png'
	depth_files2.append(cv2.imread(depth_file_name2, cv2.IMREAD_UNCHANGED))
	
	pncc_file_name = prefix + str(person_id)+ '_pncc' +'attack_cnn.png'
	pncc_files.append(cv2.imread(depth_file_name, cv2.IMREAD_UNCHANGED))
	pncc_file_name2 = prefix + str(person_id)+ '_pncc' +'attack_gbdt.png'
	pncc_files2.append(cv2.imread(depth_file_name2, cv2.IMREAD_UNCHANGED))


subplot_id = 1
person_id = 0
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
	# if subplot_id%n_file_per_person==5:
	# 	ax.imshow(pncc_files[person_id])
	# 	if person_id<3: ax.set_title('CNN Attacked')
	# if subplot_id%n_file_per_person==4:
	# 	ax.imshow(pncc_files2[person_id])
	# 	if person_id<3: ax.set_title('GBDT Attacked')

	subplot_id += 1
	if subplot_id%n_file_per_person==0:
		person_id += 1


plt.savefig('plot/comparison.png', dpi=100)
plt.show()
