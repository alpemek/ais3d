#include<iostream>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/common.h>
#include <iomanip>  
#include <string>
  
//using namespace std;  
int main(int argc, char* argv[])
{
  std::string original = "/home/dllab/kitti_object/data_object_velodyne/pcl/" + boost::to_string(argv[1])+ ".pcd" ;  
  std::string frustrum = "/home/emeka/Schreibtisch/AIS/ais3d/Final/frustrum_" + boost::to_string(argv[1]) + ".pcd" ;
  std::string segmented_base = "/home/emeka/Schreibtisch/AIS/ais3d/Final/segmented_" + boost::to_string(argv[1]) + "_" ;
  std::string segmented = "";

  pcl::PointXYZ minPt, maxPt,minPt2,maxPt2; 
  pcl::PCDReader reader;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  reader.read(original,*cloud);

 // x = strtol(argv[1], NULL, 10);
  int num_obj = strtol(argv[2], NULL, 10);


  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2(new pcl::PointCloud<pcl::PointXYZ>);
  reader.read(frustrum,*cloud2);

  
  //std::cout << cloud->points.size() << std::endl;
  //std::cout << cloud2->points.size() << std::endl;

  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer ("3D Viewer"));

  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud, 255, 0, 0);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color2(cloud2, 0, 255, 0);
  

  viewer->addPointCloud<pcl::PointXYZ>(cloud,single_color,"original");
  viewer->addPointCloud<pcl::PointXYZ>(cloud2,single_color2,"frustrum");
  
/*
  for( int i = 0; i < num_obj; i++){	
      //std::cout << cloud_dum->points.size() << std::endl;

	  segmented = segmented_base + boost::to_string(i) + ".pcd" ;
	  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_dum(new pcl::PointCloud<pcl::PointXYZ>);
	  reader.read(segmented,*cloud_dum);
	  std::string cube_id = "cube" + boost::to_string(i) ;
      std::string obj_id = "obj" + boost::to_string(i) ;
      
      pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud_dum, 0, 255, 0);
	  pcl::getMinMax3D (*cloud_dum, minPt, maxPt);
      
      viewer->addPointCloud<pcl::PointXYZ>(cloud_dum,single_color,obj_id);
	  //viewer->addCube (minPt.x, maxPt.x, minPt.y, maxPt.y,  minPt.z,  maxPt.z, 1.0, 1.0, 1.0, cube_id, 0);
	  //viewer->setRepresentationToWireframeForAllActors();
      segmented = "";
}
*/
viewer->setBackgroundColor(0.4,0.4,0.4,0);
  while (!viewer->wasStopped ())
  {
    viewer->spinOnce (100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  }
  viewer->removeAllShapes();
  viewer->removeAllPointClouds();




}


