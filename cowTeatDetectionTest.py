# import open3d as o3d
# import numpy as np


# print("->正在加载点云... ")
# pcd = o3d.io.read_point_cloud("E:\\LUCID_HLT003S-001_212900075__20230602112103597_image0.ply")
# print(pcd)

# print("->正在可视化点云")
# o3d.visualization.draw_geometries([pcd])

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)  # 提取点云
    outlier_cloud = cloud.select_by_index(ind, invert=True)  # 反向提取
 
    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])  # red
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([pcdNow,maxSet_pointcloudPart,inlier_cloud, outlier_cloud])
    

    
def get_maxPartPointCloud(labelsPara,colorsPara,pcdPara):
    lis = list()
    for colorTemp in colorsPara:        
        i = colorTemp[0]
        j = colorTemp[1]
        k = colorTemp[2]  
        listNow = [i,j,k]  
        lis.append(listNow)

    ptIndex = list()
    ptFirIndex = list()
    storeIndex  = list()

    for tempList in lis:
        firEle = tempList[0]
        secEle = tempList[1]
        thiEle = tempList[2]
        flag = bool(0)
        for tempFirIndex in ptFirIndex:
            if((firEle == tempFirIndex[0]) and (secEle == tempFirIndex[1]) and (thiEle == tempFirIndex[2])):
                flag = 1
                break
        if (flag == 0):
            ptIndex.append(tempList)
            ptFirIndex.append(tempList)

    maxSumList = list()
    maxSum = 0;
    tempMaxX = 0.0
    tempMaxY = 0.0
    tempMaxZ = 0.0
    for tempLoop in ptFirIndex:
        firEle = tempLoop[0]
        secEle = tempLoop[1]
        thiEle = tempLoop[2]
        count = 0

        for tempList in lis:
            if((firEle == tempList[0]) and (secEle == tempList[1]) and (thiEle == tempList[2])):
                count = count + 1
        if(count > maxSum):
            maxSum = count
            tempMaxX = firEle
            tempMaxY = secEle
            tempMaxZ = thiEle

        storeIndex.append(count)

    maxSumList.append(tempMaxX)
    maxSumList.append(tempMaxY)
    maxSumList.append(tempMaxZ)

    return maxSumList,lis

if __name__ == "__main__":
    # sample_ply_data = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud("E:\\LUCID_HLT003S-001_212900075__20230602112402958_image0.ply")
    downpcd = pcd.voxel_down_sample(voxel_size = 20)
    # pcd = o3d.io.read_point_cloud(sample_ply_data.path)
    # Flip it, otherwise the pointcloud will be upside down.
    downpcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        # """
        # label为每个点云对应的类别，
        # 其中label为-1的点云被认为是噪音
        # """
        labels = np.array(
            downpcd.cluster_dbscan(eps=200, min_points=10, print_progress=True))
        
    max_label = labels.max()
    partPcd = labels[0]
    print(f"point cloud has {max_label + 1} clusters")
    # 为每个类别生成颜色
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    # 将为噪音的点云设置为黑色
    colors[labels < 0] = 0
    # 给点云上色
    downpcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    #o3d.visualization.draw_geometries([downpcd])
    
    maxIndexList,lis = get_maxPartPointCloud(labels,colors,downpcd)

    countListIndex = 0
    finalMaxList = list()
    for tempList in lis:
        firEle = tempList[0]
        secEle = tempList[1]
        thiEle = tempList[2]
        if((firEle == maxIndexList[0]) and (secEle == maxIndexList[1]) and (thiEle == maxIndexList[2])):
            finalMaxList.append(countListIndex)
        countListIndex = countListIndex + 1
    
    maxSet_pointcloud=downpcd.select_by_index(finalMaxList,False)

    o3d.io.write_point_cloud("colorPCD.pcd", maxSet_pointcloud)

    labelsMaxSet = np.array(
    maxSet_pointcloud.cluster_dbscan(eps=100, min_points=10, print_progress=True))

    max_labelMaxSet = labelsMaxSet.max()
    print(f"point cloud has {max_labelMaxSet + 1} clusters")
    # 为每个类别生成颜色
    colorsMaxSet = plt.get_cmap("tab20")(labelsMaxSet / (max_labelMaxSet if max_labelMaxSet > 0 else 1))
    # 将为噪音的点云设置为黑色
    colorsMaxSet[labelsMaxSet < 0] = 0
    # 给点云上色
    maxSet_pointcloud.colors = o3d.utility.Vector3dVector(colorsMaxSet[:, :3])

    maxIndexPartList,lisPart = get_maxPartPointCloud(labelsMaxSet,colorsMaxSet,maxSet_pointcloud)

    countListIndexPart = 0
    finalMaxListPart = list()
    for tempList in lisPart:
        firEle = tempList[0]
        secEle = tempList[1]
        thiEle = tempList[2]
        if((firEle == maxIndexPartList[0]) and (secEle == maxIndexPartList[1]) and (thiEle == maxIndexPartList[2])):
            finalMaxListPart.append(countListIndexPart)
        countListIndexPart = countListIndexPart + 1
    
    maxSet_pointcloudPart=maxSet_pointcloud.select_by_index(finalMaxListPart,False)

    o3d.io.write_point_cloud("colorPCD.pcd", maxSet_pointcloudPart)

    xyz_load = np.asarray(maxSet_pointcloudPart.points)

    # list to collect local maxima
    local_maxima = []

    # distance in x / y to define region of interest around current center coordinate
    radius = 20

    for i in range(xyz_load.shape[0]):
        # radial mask with radius radius, could be beautified via numpy.linalg
        mask = np.sqrt((xyz_load[:,0] - xyz_load[i,0])**2 + (xyz_load[:,1] - xyz_load[i,1])**2) <= radius
        # if current z value equals z_max in current region of interest, append to result list
        if xyz_load[i,2] == np.max(xyz_load[mask], axis = 0)[2]:
            local_maxima.append(tuple(xyz_load[i]))


    #judge circle area  it contains surrounding specified numbers points in all directions
    #given radius

    

    #for i in range(xyz_load.shape[0]):
    #    for j in range(local_maxima):


    #maxSet_pointcloudPart.points.extend(local_maxima)

    pcdNow = o3d.geometry.PointCloud()
    pcdNow.points = o3d.utility.Vector3dVector(local_maxima)
    o3d.io.write_point_cloud("localmaximum.pcd", pcdNow)

    #pcdBoundary = o3d.t.io.read_point_cloud("colorPCD.pcd")
    #pcdBoundary.estimate_normals(max_nn=30, radius=150)

    #boundarys, mask = pcdBoundary.compute_boundary_points(150, 30)
    # TODO: not good to get size of points.
    #print(f"Detect {boundarys.point.positions.shape[0]} bnoundary points from {boundarys.point.positions.shape[0]} points.")

    #boundarys = boundarys.paint_uniform_color([1.0, 0.4, 1.0])
    #pcd = pcd.paint_uniform_color([0.5, 0.5, 0.5])
    #o3d.visualization.draw_geometries([pcdBoundary.to_legacy(), boundarys.to_legacy()],
    #                                 zoom=0.3412,
    #                                 front=[0.3257, -0.2125, -0.8795],
    #                                 lookat=[2.6172, 2.0475, 1.532],
    #                                 up=[-0.0694, -0.9768, 0.2024]
    #                             )

    cl, ind = maxSet_pointcloudPart.remove_radius_outlier(nb_points=8, radius=30)  # cl: PointCloud. ind: 处理后idx list
     # ind: idx list

    pcdNow.paint_uniform_color([0.1, 0.1, 0.7])
    maxSet_pointcloudPart.paint_uniform_color([0.1, 0.9, 0.1])

    #o3d.visualization.draw_geometries([pcdNow])
    o3d.visualization.draw_geometries([pcdNow,maxSet_pointcloudPart])
    o3d.io.write_point_cloud("colorPCD.pcd", maxSet_pointcloudPart)

    #display_inlier_outlier(maxSet_pointcloudPart, ind) 

    # print col ors  
    #o3d.visualization.draw_geometries([partPcd])


