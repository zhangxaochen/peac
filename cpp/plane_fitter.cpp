//
// Copyright 2014 Mitsubishi Electric Research Laboratories All
// Rights Reserved.
//
// Permission to use, copy and modify this software and its
// documentation without fee for educational, research and non-profit
// purposes, is hereby granted, provided that the above copyright
// notice, this paragraph, and the following three paragraphs appear
// in all copies.
//
// To request permission to incorporate this software into commercial
// products contact: Director; Mitsubishi Electric Research
// Laboratories (MERL); 201 Broadway; Cambridge, MA 02139.
//
// IN NO EVENT SHALL MERL BE LIABLE TO ANY PARTY FOR DIRECT,
// INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING
// LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS
// DOCUMENTATION, EVEN IF MERL HAS BEEN ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGES.
//
// MERL SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE. THE SOFTWARE PROVIDED HEREUNDER IS ON AN
// "AS IS" BASIS, AND MERL HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE,
// SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
//
#pragma warning(disable: 4996)
#pragma warning(disable: 4819)
#define _CRT_SECURE_NO_WARNINGS

#include <pcl/point_types.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/console/parse.h> //zc
#include <pcl/point_cloud.h> //zc: 改用 1.8之后, 似乎必须显式加此项

#include "opencv2/opencv.hpp"

#include "AHCPlaneFitter.hpp"

using ahc::utils::Timer;

#include "zcAhcUtility.h"

using ahc::PlaneSeg;
using std::vector;
using std::string;
using std::cout;
using std::endl;
using std::ofstream;

using std::set; //用于正交/平行关系表构建
using std::set_intersection;
using namespace Eigen;

// typedef Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Matrix3frm; //cube-pose 一律应为默认columnMajor
// typedef Eigen::Matrix<double, 3, 3, Eigen::RowMajor> Matrix3drm;

//const float MM2M = 1e-3;
const float cos30 = 0.8660254; //cos(30°)

bool dbg1_ = false; //用于调试输出 seg 图, 以及非内循环输出
bool dbg2_ = false; //用于调试输出 debug print

string deviceId_ = ""; //zc: 命令行参数 -dev
double fx_ = 525.5, fy_ = 525.5, cx_ = 320, cy_ = 240; //内参: 命令行 -intr
vector<string> pngFnames_;
int png_sid_ = 0; //start id 调试用

bool isQvga_ = false;
int qvgaFactor_ = 1; //if isQvga_=true, --> factor=2

//↓--TUM 默认放大 5(说是5000) 存储便于观察: http://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats#color_images_and_depth_maps
//但是我一些旧数据是原始 ushort, 故加载旧数据时, 用 -dscale 1 (默认), 加载 TUM-dataset 时, 用 -dscale 5
float dscale_ = 1;

//vector<vector<double>> cubePoses_; //弃用, 关键需要的是相机姿态增量
vector<vector<double>> cubeCandiPosesPrev_; //前【一帧】(非多帧序列)立方体的候选坐标系; //是否 vector<Affine3d> 好点? 暂时懒得改了 2016-9-27 16:11:59
static bool isFirstFrame_ = true; //if true, cubePosePrev_=curr...
int poseIdxPrev_ = 0; //cubePosesPrev_ 中用哪个idx做的立方体参考系		//【弃用】, 因为不关心立方体用哪个直角做参考系, 

string poseFn_ = ""; //相机姿态描述文件
vector<Affine3d, aligned_allocator<Affine3d>> camPoses_;
int lastValidIdx_ = -1; //camPoses_ 改成有全零无效FLAG之后, 此变量存储上次有效位置 idx //2016-12-9 00:14:51

#define SINGLE_VERT 0
#define PROCRUSTES 01


//@author zhangxaochen, 拷贝自 zc::getFnamesInDir@kinfu_app.cpp
//@brief get file names in *dir* with extension *ext*
//@param dir file directory
//@param ext file extension
vector<string> getFnamesInDir(const string &dir, const string &ext){
	namespace fs = boost::filesystem;
	//fake dir:
	//cout << fs::is_directory("a/b/c") <<endl; //false

	//fs::path dirPath(dir); //no need
	cout << "path: " << dir << endl
		<< "ext: " << ext <<endl;

	if(dir.empty() || !fs::exists(dir) || !fs::is_directory(dir)) //其实 is_directory 包含了 exists: http://stackoverflow.com/questions/2203919/boostfilesystem-exists-on-directory-path-fails-but-is-directory-is-ok
		PCL_THROW_EXCEPTION (pcl::IOException, "ZC: No valid directory given!\n");

	vector<string> res;
	fs::directory_iterator pos(dir),
		end;

	for(; pos != end; ++pos){
		if(fs::is_regular_file(pos->status()) && fs::extension(*pos) == ext){
#if BOOST_FILESYSTEM_VERSION == 3
			res.push_back(pos->path().string());
#else
			res.push_back(pos->path());
#endif
		}
	}

	if(res.empty())
		PCL_THROW_EXCEPTION(pcl::IOException, "ZC: no *" + ext + " files in current directory!\n");
	return res;
}//getFnamesInDir

//zc: OrganizedImage3D为算法调用者去实现的多态类, 没想懂: 为什么不用纯虚基类? (倒是有个 NullImage3D “示范类”) //2016-9-11 16:16:02
// pcl::PointCloud interface for our ahc::PlaneFitter
template<class PointT>
struct OrganizedImage3D {
	const pcl::PointCloud<PointT>& cloud;
	//NOTE: pcl::PointCloud from OpenNI uses meter as unit,
	//while ahc::PlaneFitter assumes mm as unit!!!
	const double unitScaleFactor;

	OrganizedImage3D(const pcl::PointCloud<PointT>& c) : cloud(c), unitScaleFactor(1000) {}
	int width() const { return cloud.width; }
	int height() const { return cloud.height; }
	bool get(const int row, const int col, double& x, double& y, double& z) const {
		const PointT& pt=cloud.at(col,row);
		x=pt.x; y=pt.y; z=pt.z;
		return pcl_isnan(z)==0; //return false if current depth is NaN
	}
};
typedef pcl::PointXYZRGBA PtType;
typedef OrganizedImage3D<PtType> RGBDImage;
typedef ahc::PlaneFitter<RGBDImage> PlaneFitter;

class MainLoop
{
protected:
	PlaneFitter pf;
	cv::Mat rgb, seg;
	bool done;

public:
	bool pause_; //zc
	pcl::OpenNIGrabber* grabber_;

public:
	MainLoop () : done(false), pause_(false) {}

	//process a new frame of point cloud
	void onNewCloud (const pcl::PointCloud<PtType>::ConstPtr &cloud)
	{
		//fill RGB
		if(rgb.empty() || rgb.rows!=cloud->height || rgb.cols!=cloud->width) {
			rgb.create(cloud->height, cloud->width, CV_8UC3);
			seg.create(cloud->height, cloud->width, CV_8UC3);
		}
// 		const PtType &tmppt = cloud->at(520,360);
// 		std::cout<<"xyz:= "<<tmppt.x<<", "<<tmppt.y<<", "<<tmppt.z<<std::endl; //观察内参是否会对3D坐标有影响, 应该有, 但似乎实际没有, bug?

		for(int i=0; i<(int)cloud->height; ++i) {
			for(int j=0; j<(int)cloud->width; ++j) {
				const PtType& p=cloud->at(j,i);
				if(!pcl_isnan(p.z)) {
					rgb.at<cv::Vec3b>(i,j)=cv::Vec3b(p.b,p.g,p.r);
				} else {
					rgb.at<cv::Vec3b>(i,j)=cv::Vec3b(255,255,255);//whiten invalid area
				}
			}
		}

		//run PlaneFitter on the current frame of point cloud
		RGBDImage rgbd(*cloud);
		Timer timer(1000);
		timer.tic();
		//pf.run(&rgbd, 0, &seg);
		vector<vector<int>> idxss;
		if(dbg1_)
			pf.run(&rgbd, &idxss, &seg);
		else
			//pf.run(&rgbd, &idxss, &seg, 0, false);
			pf.run(&rgbd, &idxss, 0, 0, false); //连 seg 也省去不要, 似乎并不省时间
		double process_ms=timer.toc();

		if(dbg1_)
			annotateLabelMat(pf.membershipImg, &seg); //release: 6ms
		//cv::imwrite("shit.png", seg); //release: 5ms

		vector<PlaneSeg> plvec; //存放各个平面参数
		timer.tic();
#if 01	//zcRefinePlsegParam 相关, 抽取成函数
		for(size_t i=0; i<idxss.size(); i++){
			vector<int> &idxs = idxss[i];
			PlaneSeg tmpSeg(rgbd, idxs);
			plvec.push_back(tmpSeg);

			if(dbg2_)
				printPlaneParams(tmpSeg);

			//zc: old/new 对比, 看refine效果 //其实差不多, 2016-9-20 15:09:57
			//PlaneSeg &oldSeg = *pf.extractedPlanes[i];
			//double *oldNorm = oldSeg.normal;
			//double *oldCen = oldSeg.center;
			//double oldCurv = oldSeg.curvature;
			//double oldMse = oldSeg.mse;

			//double *newNorm = tmpSeg.normal;
			//double *newCen = tmpSeg.center;
			//double newCurv = tmpSeg.curvature;
			//double newMse = tmpSeg.mse;
			//printf("norm--old.dotProd(new)==%f; center.dist==%f; o/nCurv=(%f, %f), o/nMse==(%f, %f)\n", dotProd(oldNorm, newNorm, 3), dist(oldCen, newCen), oldCurv, newCurv, oldMse, newMse);
		}
#else
		for(size_t i=0; i<pf.extractedPlanes.size(); i++)
			plvec.push_back(*pf.extractedPlanes[i]);

		plvec = zcRefinePlsegParam(rgbd, idxss); //决定是否 refine 平面参数
#endif	//zcRefinePlsegParam 相关, 抽取成函数

		if(dbg1_)
			timer.toc("re-calc-plane-equation:"); //debug:7ms; release:1.5ms

		//zc: 腐蚀各个面片msk, 与上面对比normal偏差
#if 0	//无论 krnl=5,7, norm-angle<0.3°,说明erode意义不大, 因此省去 2016-9-21 17:14:56
		timer.tic();
		vector<PlaneSeg> plvecErode;
		int krnlSz=7;
		cv::Mat erodeKrnl = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(krnlSz,krnlSz));
		for(size_t i=0; i<idxss.size(); i++){
			cv::Mat msk = (pf.membershipImg == i);
			cv::erode(msk, msk, erodeKrnl);
			PlaneSeg tmpSeg(rgbd, msk);
			plvecErode.push_back(tmpSeg);

			double cosNorm = dot(plvec[i].normal, tmpSeg.normal);
			printf("i= %d, cosNorm= %f, angle= %f\n", i, cosNorm, RAD2DEG(acos(cosNorm)));
		}
		timer.toc("erode-calc-plane-params:"); //release:5.4ms(erode:3ms)
#endif

		timer.tic();
		vector<vector<double>> cubeCandiPoses; //内vec必有size=12=(t3+R9), 是cube在相机坐标系的姿态, 且规定row-major; 外vec表示多个候选顶角的姿态描述

#if 01	//zcFindOrtho3tup相关, 抽取成函数
		//zc: 找两两正交的三个平面, 组成一个三元组 (可能有多组, e.g., 立方体一个竖直棱的上下顶角, 或干扰物造成), 
		//示意图见: http://codepad.org/he35YCTh
		//1. 检查所有平面, 清点正交、平行关系, 既不正交也不平行的忽略
		//size_t plcnt = plvec.size();
		size_t plcnt = idxss.size();
		double orthoThresh = 
			//0.0174524; //cos(89°), 小于此为正交 //因为依赖 pcl160, 而OpenNIGrabber用默认内参,导致点云不准, 正交面达不到此阈值
			0.0871557; //cos(85°) 放宽, //【用了雷昊内参之后, 发现非常好, 不过仍然放宽
		double paralThresh = 
			//0.999847695; //cos(1°), 大于此为平行
			0.996194698; //cos(5°) 放宽

		//vector<vector<int>> orthoMap(plcnt); //正交关系表, 理论上只填充上三角, 即最后一个vec应是空; 【其实 vector<set<int>> 更合理, 暂不改
		//vector<vector<int>> paralMap(plcnt); //平行关系表
		//for(size_t i=0; i<plcnt; i++){
		//	PlaneSeg &pl_i = plvec[i];
		//	double *norm_i = pl_i.normal;

		//	for(size_t j=i+1; j<plcnt; j++){ //从 i+1, 只看上三角
		//		PlaneSeg &pl_j = plvec[j];
		//		double *norm_j = pl_j.normal;
		//		double cosNorm = dot(norm_i, norm_j, 3); //因 |a|=|b|=1, 故直接 cos(a,b)=a.dot(b)

		//		if(cosNorm < orthoThresh)
		//			orthoMap[i].push_back(j);
		//		if(cosNorm > paralThresh)
		//			paralMap[i].push_back(j);
		//	}//for-j
		//}//for-i

		////2. 在每个 orthoMap[i] 中, 若还有正交的, 则找到一个三元组!
		//for(size_t i=0; i<orthoMap.size(); i++){
		//	vector<int> &ortho_i = orthoMap[i]; //与id=i正交的平面id们
		//	for(size_t j=0; j<ortho_i.size(); j++){
		//		int oij = ortho_i[j];
		//		vector<int> &ortho_j = orthoMap[oij]; //固定i,j之后
		//		for(size_t k=j+1; k<ortho_i.size(); k++){

		//		}
		//	}
		//}

		vector<set<int>> orthoMap(plcnt);  //正交关系表, 理论上只填充上三角, 即最后一个vec应是空; 【其实 vector<set<int>> 更合理, 暂不改
		vector<vector<int>> ortho3tuples; //找到的三元组放在这, 内vec 故意用 vec 不用 set, 更易索引; 内vec必然size=3
		vector<set<int>> paralMap(plcnt); //平行关系表
		for(size_t i=0; i<plcnt; i++){
			PlaneSeg &pl_i = plvec[i];
			double *norm_i = pl_i.normal;
			for(size_t j=i+1; j<plcnt; j++){ //从 i+1, 只看上三角
				PlaneSeg &pl_j = plvec[j];
				double *norm_j = pl_j.normal;
				double cosNorm = dotProd(norm_i, norm_j, 3); //因 |a|=|b|=1, 故直接 cos(a,b)=a.dot(b)

				if(dbg2_)
					printf("i,j=(%u,%u), cosNorm=%f; angle=%f\n", i, j, cosNorm, RAD2DEG(acos(cosNorm)));

				if(abs(cosNorm) < orthoThresh)
					orthoMap[i].insert(j);
				if(abs(cosNorm) > paralThresh)
					paralMap[i].insert(j);
			}//for-j
		}//for-i

		//2. 在每个 orthoMap[i] 中, 若还有正交的, 则找到一个三元组! //理论上【已知立方体】上只能找到两个三元组, 若>2, 则要通过平行面间距做排除
		for(size_t i=0; i<orthoMap.size(); i++){
			set<int> &ortho_i = orthoMap[i]; //与id=i正交的平面id们
			if(ortho_i.size() < 2) //若 <2, 则构不成三元组
				continue;
			set<int>::const_iterator iter_j = ortho_i.begin();
			for(; iter_j!=ortho_i.end(); iter_j++){
				int idx_j = *iter_j;
				set<int> &ortho_j = orthoMap[idx_j]; //即将在此找 idx_k, 若找到, 则成一个三元组
				if(ortho_j.size() == 0)
					continue;
				//set<int>::const_iterator iter_k = iter_j + 1; //×, 无 '+'
				//set<int>::const_iterator iter_k = iter_j; iter_k++; //√, 不过原始
				set<int>::const_iterator iter_k = std::next(iter_j);
				for(; iter_k!=ortho_i.end(); iter_k++){
					int idx_k = *iter_k;
					if(ortho_j.count(idx_k)){ //找到三元组
						vector<int> tuple3;
						tuple3.push_back(i);
						tuple3.push_back(idx_j);
						tuple3.push_back(idx_k);
						ortho3tuples.push_back(tuple3);
					}
				}//for-iter_k
			}//for-iter_j
		}//for-i

		//3. 在 ortho3tuples 可能存在【假的】三元组, 判定依据: 三面实际相邻才算三元组-->进化为: 【放弃】三面交点(3D)在labelMat(2D)上某len邻域内有三种有效label【放弃】
		//判定依据改为: 邻域label-set 包含 tuple3-vec (虽然一个 set 一个 vector, 用 std::includes 算法) //2016-12-7 20:55:25
		vector<vector<int>> tmp3tuples;
		for(size_t i=0; i<ortho3tuples.size(); i++){
			vector<int> &tuple3 = ortho3tuples[i];
			//1. 取三面, 构造 Ax=b 的 [A|b] 增广矩阵; 列主元高斯, 求解三面顶点
			vector<vector<double>> matA;
			vector<double> matb;
			for (int ii=0; ii<3; ii++){
				int plIdx = tuple3[ii];
				PlaneSeg &plseg = plvec[plIdx];
				//平面参数ABCD: (ABC)=normal; D= -dot(normal, center) //注意负号, b[i]=-D=dot...
				vector<double> tmpRow(plseg.normal, plseg.normal+3);
				double b_i = dotProd(plseg.normal, plseg.center);
				tmpRow.push_back(b_i); //系数矩阵一行, 包含 Ai|bi
				matA.push_back(tmpRow);
			}
			vector<double> vertx; //三面交点, 方程组的解; 尺度是不是米(m)啊？ √
			RGauss(matA, vertx);

			//2. 顶点的像素邻域内是否有三种有效label
			//3D->2D像素点, 是int, 不追求精度
			int u = (vertx[0] * fx_) / vertx[2] + cx_,
				v = (vertx[1] * fy_) / vertx[2] + cy_;

			int winSz = 20 / qvgaFactor_; //邻域窗口长度
			cv::Rect tmpRoi(u - winSz/2, v - winSz/2, winSz, winSz);
			if(tmpRoi != (tmpRoi & cv::Rect(0,0, seg.cols, seg.rows)) ) //若邻域不在图像范围内, 忽略
				continue;
			//else: 否则, 若整个邻域小方框都在图像内, 继续
			if(dbg1_){
				cv::circle(seg, cv::Point(u, v), 2, 255, 1); //蓝小圆圈
				cv::circle(seg, cv::Point(u, v), 7, cv::Scalar(0,0,255), 2); //红大圆圈, 同心圆, 调试观察便利
			}

			cv::Mat vertxNbrLmat(pf.membershipImg, tmpRoi); //邻域 label-mat
			vertxNbrLmat = vertxNbrLmat.clone(); //clone 能解决set(labelMat) 出错问题吗? √ 能!
					//记不清了, 应该是说, 若不 clone, 则roi仅是view而非连续内存, 取 [data, data+size] 时会取到原 Mat 一行片段(如 1*16), 而非真正需要的方块(如4*4) //2016-12-28 11:22:14
			//cout<<"vertxNbrLmat:\n"<<vertxNbrLmat<<endl;
			//cv::Mat dbgRoiMat(seg, tmpRoi); //调试观察小区域

			int *vertxNbrLmatData = (int*)vertxNbrLmat.data; //label mat raw data
			set<int> nbrLabelSet(vertxNbrLmatData, vertxNbrLmatData + winSz * winSz);
			//int posCnt = 0; //邻域 label>0 统计量
			//for(set<int>::const_iterator it = nbrLabelSet.begin(); it != nbrLabelSet.end(); it++){
			//	if(*it >= 0) //0也是有效label
			//		posCnt++;
			//}
			if(std::includes(nbrLabelSet.begin(), nbrLabelSet.end(), tuple3.begin(), tuple3.end()) ){ //已验证, 效果很好 2016-12-9 00:09:40
			//if(posCnt >= 3){ //认定为真实正交三邻面三元组
				if(dbg1_){
					cv::circle(seg, cv::Point(u, v), 2, 255, 1); //蓝小圆圈
					cv::circle(seg, cv::Point(u, v), 7, cv::Scalar(0,255,0), 2); //绿大圆圈, 同心圆, 调试观察便利, //表示筛选最终定下的顶角
				}

				tmp3tuples.push_back(tuple3);
				cubeCandiPoses.push_back(vertx); //先把(R,t)的t填充; 之后下面 cubePoses 不要 push, 要在每行 .insert(.end, dat, dat+3);
			}
		}//for-ortho3tuples
		ortho3tuples.swap(tmp3tuples);

		if(cubeCandiPoses.size() == 2){
			Vector3d pt0(cubeCandiPoses[0].data());
			Vector3d pt1(cubeCandiPoses[1].data());
			
			printf("上下两顶角距离：%f\n", (pt0 - pt1).norm());
		}

		//4. 根据邻面、交线定位立方体; 若一帧有多个三元组，则都计算，故意冗余，以免下一帧三元组减少而跟丢
		for(size_t i=0; i<ortho3tuples.size(); i++){
			vector<int> &tuple3 = ortho3tuples[i];
			//vector<double> pose_i; //待生成的候选姿态描述子之一; 此处仅 R; 前面已经push过 t(=vertx)
			vector<double> ax3orig; //初始三轴, 不完美正交
			ax3orig.reserve(9);

			//4.1. 主轴策略: 三邻面未必正交, 要用【施密特正交化】, 要先选主轴, 后两轴依附于此; 主轴未必是cube的xyz哪一轴, 确定xyz要根据轴长度
#if 0	//v1: 法向与Z轴(非O->plCenter视线)夹角小(即abs(normal.z)大的)的做主轴; 第二小的做第二轴; 1,2叉积做第三轴
			//找主轴：
			double maxAbsZ = 0, minorAbsZ = 0;
			int mainPlIdx = -1, minorPlIdx = -1;
			for(size_t j=0; j<3; j++){
				int plIdx = tuple3[j];
				PlaneSeg &plseg = plvec[plIdx];
				double absz = abs(plseg.normal[2]);
				if(absz > maxAbsZ){
					maxAbsZ = absz;
					//minorPlIdx = mainPlIdx; //第二轴用上次的主轴 //逻辑错
					mainPlIdx = plIdx;
				}
				if(absz < maxAbsZ && absz > minorAbsZ){
					minorAbsZ = absz;
					minorPlIdx = plIdx;
				}
			}//for-tuple3

			double *mainAxis = plvec[mainPlIdx].normal;
			double *minorAxis = plvec[minorPlIdx].normal;

#if 0	//v1.1: 用三个数组
			double *ax1 = new double[3];
			double *ax2 = new double[3];
			double *ax3 = new double[3];
			schmidtOrtho(mainAxis, minorAxis, ax1, ax2, 3);
			crossProd(ax1, ax2, ax3); //因为叉积, ax1,2,3自然呈右手系, 是旋转矩阵
			pose_i.insert(pose_i.end(), ax1, ax1 + 3);
			pose_i.insert(pose_i.end(), ax2, ax2 + 3);
			pose_i.insert(pose_i.end(), ax3, ax3 + 3);
#elif 1	//v1.2: 用【一个】数组存储
			//double *axs = new double[9]; //何必new呢?
			double axs[9];
			schmidtOrtho(mainAxis, minorAxis, axs, axs+3, 3);
			crossProd(axs, axs+3, axs+6); //生成第三轴
			pose_i.insert(pose_i.end(), axs, axs+9);
#endif	//施密特

#elif 1	//v2: 最优正交化, 见: https://www.evernote.com/shard/s399/nl/67976577/48135b5e-7209-47c1-9330-934ac4fee823
#if 01	//v2.1 三面【法向】做轴, 不正交没关系
			for(size_t kk=0; kk<3; kk++){
				double *pl_k_norm = plvec[tuple3[kk]].normal;
				ax3orig.insert(ax3orig.end(), pl_k_norm, pl_k_norm+3);
			}
#elif 1	//v2.2 三面【交线】做轴,	//最终相机姿态竟然与上面【法向做轴】全等!!! 暂未推导, 下面测试代码已验证 -↓
			//交线只需要管方向, 因为顶点之前已经 RGauss 求出来; 两两法向叉积即可
			for(size_t ii=0; ii<3; ii++){
				double *norm_i = plvec[tuple3[ii]].normal;
				for (size_t jj=ii+1; jj<3; jj++){
					double *norm_j= plvec[tuple3[jj]].normal;
					double intersLine[3];
					crossProd(norm_i, norm_j, intersLine);
					ax3orig.insert(ax3orig.end(), intersLine, intersLine+3);
				}
			}
#endif //法向/交线 谁做初始轴

			//Matrix3d ttmp = Map<Matrix3d>(ax3orig); //编译错, 必须 .data()
			//JacobiSVD<Matrix3d> svd(Map<Matrix3d>(ax3orig)); //错, 虽然编译过, 但是假的, 注意 warning C4930: prototyped function not called
			//Matrix3d tmp = Map<Matrix3d>(ax3orig.data()); //ok
			//JacobiSVD<Matrix3d> svd(tmp, ComputeThinU | ComputeThinV); //运行错, JacobiSVD: thin U and V are only available when your matrix has a dynamic number of columns
			JacobiSVD<Matrix3d> svd(Map<Matrix3d>(ax3orig.data()), ComputeFullU | ComputeFullV);
			Matrix3d svdU = svd.matrixU();
			Matrix3d svdV = svd.matrixV();
			Matrix3d orthoAxs = svdU * svdV.transpose(); //这里得到关于 ax3orig 的最优化正交基, det=±1, 不确保是旋转矩阵
			double *axs = orthoAxs.data();
			//pose_i.insert(pose_i.end(), axs, axs+9);

			//+++++++++++++++tmp: 查找分析与 【交线做轴】全等的原因	//答:最优化之后, 两组正交基确实全等
// 			{
// 			vector<double> tmp;
// 			for(size_t ii=0; ii<3; ii++){
// 				double *norm_i = plvec[tuple3[ii]].normal;
// 				for (size_t jj=ii+1; jj<3; jj++){
// 					double *norm_j= plvec[tuple3[jj]].normal;
// 					double intersLine[3];
// 					crossProd(norm_i, norm_j, intersLine);
// 					tmp.insert(tmp.end(), intersLine, intersLine+3);
// 				}
// 			}
// 			JacobiSVD<Matrix3d> svd(Map<Matrix3d>(tmp.data()), ComputeFullU | ComputeFullV);
// 			Matrix3d svdU = svd.matrixU();
// 			Matrix3d svdV = svd.matrixV();
// 			Matrix3d orthoAxs = svdU * svdV.transpose(); //这里得到关于 ax3orig 的最优化正交基, det=±1, 不确保是旋转矩阵
// 			double *axs = orthoAxs.data();
// 			}
		
#endif //几种不同正交化方式

			//v3: 不正交化, 按 procrustes问题处理: https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
#if PROCRUSTES
			axs = ax3orig.data();
#endif //PROCRUSTES

			//cubeCandiPoses[i].insert(cubeCandiPoses[i].end(), pose_i.begin(), pose_i.end());
			cubeCandiPoses[i].insert(cubeCandiPoses[i].end(), axs, axs+9); //改用 raw指针试试玩
		}//for-ortho3tuples
#else
		zcFindOrtho3tup(plvec, pf.membershipImg, fx_, fy_, cx_, cy_, cubeCandiPoses, seg);
#endif	//zcFindOrtho3tup相关, 抽取成函数
		//---------------目前逻辑: 
		//1. 允许初始帧找不到正交顶角; 
		//2. 凡找不到的, ① 此处 tvec=(1e3,1e3,1e3)极大值, ② 并且输出 csv 中全零, 作为无效标识
		//3. 从找到顶角的第一帧开始, 填有效值 (1.5,1.5, -0.3), 并且此相机坐标系 作为之后所有帧的全局参考系
		//4. 若中间某(i)帧跟丢, 重新第(j)找到之后, 其姿态参照(i)来计算

		//当前帧是否找到至少一个顶角？ //2016-12-8 14:44:29
		bool isFindOrthoCorner = (cubeCandiPoses.size() != 0);
		if(!isFindOrthoCorner){ //若无正交顶角, 本程序中姿态置为无效值
			cout << "NO-CORNER-FOUND~" << endl;
			Affine3d invalidPose;
			invalidPose.linear() = Matrix3d::Identity();
			invalidPose.translation() << 1e3, 1e3, 1e3; //设置 t 非常大作为无效值 FLAG

			camPoses_.push_back(invalidPose);
		}
		else{ //isFindOrthoCorner==true
			cout << "isFindOrthoCorner==true" << endl;

			if(isFirstFrame_){
				isFirstFrame_ = false;

				Affine3d initPose;
				//init_Rcam_ = Eigen::Matrix3f::Identity ();// * AngleAxisf(-30.f/180*3.1415926, Vector3f::UnitX());
				//init_tcam_ = volume_size * 0.5f - Vector3f (0, 0, volume_size (2) / 2 * 1.2f);
				//↑--拷贝自@kinfu.cpp
				initPose.linear() = Matrix3d::Identity();
				initPose.translation() << 1.5, 1.5, -0.3;

				camPoses_.push_back(initPose);
			}
			else{ //isFirstFrame_==false
				//+++++++++++++++TODO
				//不止与 cubePosesPrev_[poseIdxPrev_] 对比, 因为前一帧有的, 当前帧可能跟丢; 要遍历, 找prev&curr匹配的直角
				size_t i_match, //curr
					j_match; //prev
				
				//上面 i/j_match 是 单顶角时用的, 改用多顶角了, 则改用 match-ind-vec, 但仍保留上面	//2016-12-14 15:56:00
				vector<size_t> i_match_vec, j_match_vec; //此二者 size 必然相同
				
				bool foundMatch = false;

				vector<double> multiRmats_; //若多顶角定位 (!SINGLE_VERT), 则用此, 长度=9*k; //不要做全局变量, 会随循环积累
				vector<double> multiRmatsPrev_;

				for(size_t i=0; i<cubeCandiPoses.size(); i++){
					vector<double> &rt = cubeCandiPoses[i];
					double *pRcurr = rt.data() + 3;

					//size_t j;
					for(size_t j=0; j<cubeCandiPosesPrev_.size(); j++){
						vector<double> &rtPrev = cubeCandiPosesPrev_[j];
						double vertsDist = dist(rt.data(), rtPrev.data()); //poses[i]-posesPrev[j] 顶点t间距, data()取指针
						if(vertsDist < 0.05){ //单位米(非毫米), 5cm; 若前后两帧某两候选点距离小于阈值, 则退出循环查找;
							foundMatch = true;

							//单顶角策略 //放弃 //2016-12-14 15:57:12
							i_match = i;
							j_match = j;

							//多顶角策略
							i_match_vec.push_back(i);
							j_match_vec.push_back(j);

							//顶点相近, 但可能不同轴, 存在置换式旋转, 需要对curr重排序; 此处外循环是 prev
							vector<double> tmpRcurr; //size=9
							double *pRprev = rtPrev.data() + 3;
							for(size_t axi=0; axi<3; axi++){
								double *pAxi = pRprev + axi * 3;
								for(size_t axj=0; axj<3; axj++){
									double *pAxj = pRcurr + axj * 3;
									double currPrevCos = dotProd(pAxi, pAxj, 3);
									if(abs(currPrevCos) > cos30){ //此内循环必有一次 if true
										tmpRcurr.insert(tmpRcurr.end(), pAxj, pAxj + 3);
										if(currPrevCos < -cos30){
											for(size_t kk=0, idx=tmpRcurr.size()-1; kk<3; kk++, idx--)
												tmpRcurr[idx] *= -1;
										}
										break;
									}
								}
							}
							std::copy(tmpRcurr.begin(), tmpRcurr.end(), rt.begin()+3);

#if SINGLE_VERT
							break;
#else					//不要break, 由单顶角定位, 改成多顶角定位
							multiRmats_.insert(multiRmats_.end(), rt.begin()+3, rt.end());
							multiRmatsPrev_.insert(multiRmatsPrev_.end(), rtPrev.begin()+3, rtPrev.end());
							//break;
#endif //SINGLE_VERT

						}
					}
#if SINGLE_VERT
					if(foundMatch)
						break;
					//#else
#endif //SINGLE_VERT
				}
				//assert(foundMatch); //若无 match, 说明运动过快; 或者代码有bug, 需要调试
				if(!foundMatch)
					return;

				if(poseIdxPrev_ != j_match){ //若当前帧与前一帧的匹配参考系【不是】前一帧(以及初始帧)采用的参考系 //×, 不管这些
					//DEPRECATED...
				}

				//求解delta(R,t): 是相机坐标系, 不是立方体的
				//Affine3d cuPose, cuPosePrev, //第 (i-1), i 帧; 不关心参考哪个直角坐标系, 只关心前后是关于【同一个】坐标系
				//deltaPose; //后一帧相对之前姿态增量

#if SINGLE_VERT
				Matrix3d cuRi_1, //R(i-1)
					cuRi; //R(i)
				Vector3d cuTi_1, //t(i-1)
					cuTi; //t(i)

				//虽然求解 R 已采用多顶角, 但是 t 仍然仅选了一个(循环尾)顶角做参考, 感觉不完美, 暂时不改了 //2016-12-8 23:46:21
				vector<double> &rt = cubeCandiPoses[i_match];
				cuTi = Map<Vector3d>(rt.data());
				cuRi = Map<Matrix3d>(rt.data()+3);

				vector<double> &rtPrev = cubeCandiPosesPrev_[j_match];
				cuTi_1 = Map<Vector3d>(rtPrev.data());
				cuRi_1 = Map<Matrix3d>(rtPrev.data()+3);
#else //multi-vert //2016-12-14 16:40:32
				//这里俩 t 是“N顶角重心”, 即 sum(verts)/N, 需要初始化 000
				Vector3d cuTi_1(0,0,0), //t(i-1)
					cuTi(0,0,0); //t(i)
				size_t nMatch = i_match_vec.size();
				for(size_t i=0; i<nMatch; i++){
					size_t indi = i_match_vec[i];
					size_t indj = j_match_vec[i]; //相当于上面的 i/j_match
					vector<double> &rt = cubeCandiPoses[indi];
					vector<double> &rtPrev = cubeCandiPosesPrev_[indj];
					cuTi += Vector3d(rt.data());
					cuTi_1 += Vector3d(rtPrev.data());
				}
				cuTi /= nMatch;
				cuTi_1 /= nMatch;
#endif //SINGLE_VERT

#if 0		//放弃, 需要从前面内存层上就置换好	//其实也可以此段中反处理内存层, 暂不
				Matrix3d dR = cuRi_1 * cuRi.transpose(); //R(i-1)*Ri^(-1)
				//理论上 dR约等于Eye, 但实际上因为坐标轴主轴策略, 可能存在置换轴, 应对 Ri 预先做对应旋转, 使其约等于 R(i-1):
				dR = (dR.array() > 0.9).select(1, dR); //暂定, 这里用 0.9 划分 1/0 不完全适合; 理论上此时 dR应只含 1,0, 为置换轴旋转矩阵
				dR = ((dR.array() >= -0.9) * (dR.array() <= 0.9)).select(0, dR);
				dR = (dR.array() < -0.9).select(-1, dR);

				cuRi = dR * cuRi;
				dR = cuRi_1 * cuRi.transpose();
#endif

#if PROCRUSTES //已验证, 与 最优化正交化 误差 3e-5m=0.03mm, 几乎全等; 但此方式有 "多顶角共同优化" 的潜力 //2016-10-3 01:21:15
				//tmp-test: 测试 procrustes 方法; 上面刻意 cubeCandiPoses 存的是三法向, 没做正交化 //2016-10-3 00:12:28
				//对 R=argmin|R*A-B|, M=B*A.T; 这里 A=i, B=i-1
#if SINGLE_VERT
				//Matrix3d M = cuRi * cuRi_1.transpose(); //×
				Matrix3d M = cuRi_1 * cuRi.transpose(); //√, 暂时没仔细推
#else //!SINGLE_VERT //multi-vert 多顶角定位
				assert(multiRmats_.size() != 0); //此段内必不为零
				//Matrix3Xd rmats = Map<Matrix3Xd>(multiRmats_.data()); //报错: YOU_CALLED_A_FIXED_SIZE_METHOD_ON_A_DYNAMIC_SIZE_MATRIX_OR_VECTOR
				Matrix3Xd rmats = Map<Matrix3Xd>(multiRmats_.data(), 3, multiRmats_.size()/3); //3*3*m 横向排列3*3矩阵m个
				Matrix3Xd rmatsPrev = Map<Matrix3Xd>(multiRmatsPrev_.data(), 3, multiRmatsPrev_.size()/3);
				Matrix3d M = rmatsPrev * rmats.transpose();

#endif //!SINGLE_VERT

				// 			JacobiSVD<Matrix3d> svd(Map<Matrix3d>(ax3orig.data()), ComputeFullU | ComputeFullV);
				JacobiSVD<Matrix3d> svd(M, ComputeFullU | ComputeFullV);
				Matrix3d svdU = svd.matrixU();
				Matrix3d svdV = svd.matrixV();
				Matrix3d dR = svdU * svdV.transpose(); //这里是旋转矩阵了, 因为 cuRi 与 cuRi1 同轴, 只可能旋转配准
#endif //PROCRUSTES

				// #if !SINGLE_VERT //multi-vert 多顶角定位
				// 			assert(multiRmats_.size() != 0); //此段内必不为零
				// 			//Matrix3Xd rmats = Map<Matrix3Xd>(multiRmats_.data()); //报错: YOU_CALLED_A_FIXED_SIZE_METHOD_ON_A_DYNAMIC_SIZE_MATRIX_OR_VECTOR
				// 			Matrix3Xd rmats = Map<Matrix3Xd>(multiRmats_.data(), 3, multiRmats_.size()/3); //3*3*m 横向排列3*3矩阵m个
				// 			/*Matrix3d */M = rmats * rmats.transpose();
				// 			//JacobiSVD<Matrix3d> svd2(M, ComputeFullU | ComputeFullV);
				// 			svd = JacobiSVD<Matrix3d>(M, ComputeFullU | ComputeFullV);
				// 			/*Matrix3d */svdU = svd.matrixU();
				// 			/*Matrix3d */svdV = svd.matrixV();
				// 			dR = svdU * svdV.transpose(); //这里是旋转矩阵了, 因为 cuRi 与 cuRi1 同轴, 只可能旋转配准
				// #endif //!SINGLE_VERT

				Vector3d dT = -dR * cuTi + cuTi_1; //-dR*ti+t(i-1); 平移 tvec 一直用三面交点, 不管是否正交化轴

				//Affine3d &camPosePrev = camPoses_.back(); //camPoses_.back 目前可能是无效全零 FLAG, 所以不该这样用了 //2016-12-9 00:03:17
				Affine3d &camPosePrev = camPoses_[lastValidIdx_];
				Matrix3d Ri1 = camPosePrev.linear();
				Vector3d ti1 = camPosePrev.translation();

				//dR求解之后, 计算相机姿态;
				Affine3d camPoseCurr;
				camPoseCurr.linear() = Ri1 * dR; //Ri=R(i-1)*ΔRi
				camPoseCurr.translation() = Ri1 * dT + ti1; //ti=R(i-1)*Δti + t(i-1)

				camPoses_.push_back(camPoseCurr);
			}//isFirstFrame_==false
			cubeCandiPosesPrev_ = cubeCandiPoses;
			lastValidIdx_ = camPoses_.size() - 1;
		}//isFindOrthoCorner==true

		//blend segmentation with rgb
		//cv::cvtColor(seg,seg,CV_RGB2BGR);
		//seg=(rgb+seg)/2.0;
		
		//show frame rate
		std::stringstream stext;
		stext<<"Frame Rate: "<<(1000.0/process_ms)<<"Hz";
		cv::putText(seg, stext.str(), cv::Point(15,15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255,1));

		//zc: imshow 不在UI主线程可能导致下述报错:
		//1. QCoreApplication::sendPostedEvents: Cannot send posted events for objects in another thread
		//2. QObject::moveToThread: Widgets cannot be moved to a new thread
		//3. QPixmap: It is not safe to use pixmaps outside the GUI thread
// 		cv::imshow("rgb", rgb);
// 		cv::imshow("seg", seg);
	}//onNewCloud

	//start the main loop
	void run ()
	{
		if(deviceId_ != ""){
			//pcl::Grabber* grabber = new pcl::OpenNIGrabber();
			//pcl::Grabber* grabber = new pcl::OpenNIGrabber("D:/Users/zhangxaochen/Documents/geo-cube399/00f80.oni");
			//pcl::Grabber* grabber = new pcl::OpenNIGrabber(deviceId_);
			grabber_ = new pcl::OpenNIGrabber(deviceId_);
			//grabber_->setDepthCameraIntrinsics(599.1f, 594.6f, 325.4, 252.1); //有效, 但注意正确设置 path, 用 pcl180, 而非 160; 但是对两平面垂直关系检测(有效)影响不大, 于是放宽垂直判定阈值
			// 		grabber_->setDepthCameraIntrinsics(579.267,585.016,311.056,242.254); //雷昊内参
			// 		grabber_->setDepthCameraIntrinsics(582.142,582.142,316.161,243.907); //Herrera内参
			grabber_->setDepthCameraIntrinsics(fx_, fy_, cx_, cy_);

			boost::function<void (const pcl::PointCloud<PtType>::ConstPtr&)> f =
				boost::bind (&MainLoop::onNewCloud, this, _1);

			grabber_->registerCallback(f);

			//grabbing loop
			grabber_->start();

			cv::namedWindow("rgb");
			cv::namedWindow("seg");
			cv::namedWindow("control", cv::WINDOW_NORMAL);

			int mergeMSETol=(int)pf.params.stdTol_merge,
				minSupport=(int)pf.minSupport,
				doRefine=(int)pf.doRefine;
			cv::createTrackbar("epsilon","control", &mergeMSETol, (int)pf.params.stdTol_merge*2);
			cv::createTrackbar("T_{NUM}","control", &minSupport, pf.minSupport*5);
			cv::createTrackbar("Refine On","control", &doRefine, 1);
			cv::createTrackbar("windowHeight","control", &pf.windowHeight, 2*pf.windowHeight);
			cv::createTrackbar("windowWidth","control", &pf.windowWidth, 2*pf.windowWidth);

			minSupport=0;

			//zc: play pf params
			pf.drawCoarseBorder = true;

			//GUI loop
			while (!done)
			{
				pf.params.stdTol_merge=(double)mergeMSETol;
				pf.minSupport=minSupport;
				pf.doRefine=doRefine!=0;

				//zc: imshow更新画面放到主线程
				cv::imshow("rgb", rgb);
				cv::imshow("seg", seg);

				//onKey(cv::waitKey(1000));
				onKey(cv::waitKey(this->pause_?0:1));
			}

			grabber_->stop();
		}

		if(0 != pngFnames_.size()){
			//拷贝:
			cv::namedWindow("rgb");
			cv::namedWindow("seg");
			cv::namedWindow("control", cv::WINDOW_NORMAL);

			if(isQvga_){
				pf.minSupport /= (qvgaFactor_*qvgaFactor_);
				pf.windowHeight /= qvgaFactor_;
				pf.windowWidth /= qvgaFactor_;
			}

			int mergeMSETol=(int)pf.params.stdTol_merge,
				minSupport=(int)pf.minSupport,
				doRefine=(int)pf.doRefine;
			cv::createTrackbar("epsilon","control", &mergeMSETol, (int)pf.params.stdTol_merge*2);
			cv::createTrackbar("T_{NUM}","control", &minSupport, pf.minSupport*5);
			cv::createTrackbar("Refine On","control", &doRefine, 1);
			cv::createTrackbar("windowHeight","control", &pf.windowHeight, 2*pf.windowHeight);
			cv::createTrackbar("windowWidth","control", &pf.windowWidth, 2*pf.windowWidth);

			minSupport=0;
			pf.minSupport = minSupport; //刻意置零

			//zc: play pf params
			pf.drawCoarseBorder = true;

			//cloud只构造一次:
			pcl::PointCloud<PtType>::Ptr pngCloud(new pcl::PointCloud<PtType>);
			pngCloud->is_dense = false; //false==有 nan
			pngCloud->width = 640 / qvgaFactor_;
			pngCloud->height = 480 / qvgaFactor_;
			//pngCloud->reserve(pngCloud->width * pngCloud->height); //预分配内存, 有效: 75->50ms
			pngCloud->resize(pngCloud->width * pngCloud->height); //预分配内存, 有效: 75->50ms

			//对每张 dmap-png：
			for(size_t i=png_sid_; i<pngFnames_.size() && !this->done; i++){ //if done -> break
				string &fn = pngFnames_[i];
				printf("---------------png idx:= %d\n%s\n", i, fn.c_str());

				//mat->cloud
				Timer timer(1000);
				timer.tic();
				cv::Mat dmat = cv::imread(fn, cv::IMREAD_UNCHANGED); //必须是 640*480
				dmat.convertTo(dmat, dmat.type(), dscale_);

				if(isQvga_)
					cv::pyrDown(dmat, dmat);

				timer.toctic("cv::imread: ");
				for(size_t iy=0; iy<dmat.rows; iy++){
					for(size_t ix=0; ix<dmat.cols; ix++){
						PtType pt;
						pt.r = pt.g = pt.b = 255; //因为只关心 depth-cloud, 所以 rgb 总是 fake
						ushort z = dmat.at<ushort>(iy, ix);
						if(z == 0){
							pt.x = pt.y = pt.z = std::numeric_limits<float>::quiet_NaN ();
						}
						else{
							float z_m = z * MM2M; //毫米->米(m); cloud 标准单位
							pt.x = (ix - cx_) / fx_ * z_m;
							pt.y = (iy - cy_) / fy_ * z_m;
							pt.z = z_m;
						}
						//pngCloud->points.push_back(pt); //reserve
						pngCloud->points[iy * dmat.cols + ix] = pt;
					}//for-ix
				}//for-iy
				timer.toc("mat->cloud:= ");

				this->onNewCloud(pngCloud);
				timer.toc("onNewCloud:= ");

				pf.params.stdTol_merge=(double)mergeMSETol;
				pf.minSupport=minSupport;
				pf.doRefine=doRefine!=0;

				//zc: imshow更新画面放到主线程
				cv::imshow("rgb", rgb);
				cv::imshow("seg", seg);

				//onKey(cv::waitKey(1000));
				onKey(cv::waitKey(this->pause_?0:1));
			}//for-pngFnames
			
			if(poseFn_ != "") //png 循环结束后, 若命令行 "-sp", 则存相机姿态描述文件 csv 
				processPoses(poseFn_.c_str(), camPoses_);
		}

	}//MainLoop-run

	//handle keyboard commands
	void onKey(const unsigned char key)
	{
		static bool stop=false;
		switch(key) {
		case 'q': case 27:
			this->done=true;
			break;
		case ' ':
			this->pause_ = !this->pause_;
// 			if(pause_)
// 				this->grabber_->stop();
// 			else //pause_==false
// 				this->grabber_->start();
			break;
		case 's':
			this->pause_ = true;
// 			this->grabber_->stop();
			break;
		}
	}
};

//int main ()
int main (int argc, char* argv[])
{
	MainLoop loop;

	//可能用到 MainLoop 实例：
	using namespace pcl::console;
	parse_argument(argc, argv, "-dev", deviceId_); //zc
	vector<double> depth_intrinsics;
	if(parse_x_arguments(argc, argv, "-intr", depth_intrinsics) > 0){
		fx_ = depth_intrinsics[0];
		fy_ = depth_intrinsics[1];
		cx_ = depth_intrinsics[2];
		cy_ = depth_intrinsics[3];
	}

	string png_dir;
	if(parse_argument(argc, argv, "-png_dir", png_dir) > 0){//若要用 dmap-png
		loop.pause_ = true; //若用 png, 默认暂停
		pngFnames_ = getFnamesInDir(png_dir, ".png");
		if(0 == pngFnames_.size()){
			std::cout << "No PNG files found in folder: " << png_dir << std::endl;
			return -1;
		}
		std::sort(pngFnames_.begin(), pngFnames_.end());

		parse_argument(argc, argv, "-png_sid", png_sid_);
	}

	parse_argument(argc, argv, "-sp", poseFn_);
	//parse_argument(argc, argv, "-dbg1", dbg1_);
	dbg1_ = find_switch(argc, argv, "-dbg1");
	dbg2_ = find_switch(argc, argv, "-dbg2");
	if(dbg2_) //dbg2包含dbg1
		dbg1_ = true;

	isQvga_ = find_switch(argc, argv, "-qvga");
	if(isQvga_){
		qvgaFactor_ = 2;
		fx_ /= qvgaFactor_;
		fy_ /= qvgaFactor_;
		cx_ /= qvgaFactor_;
		cy_ /= qvgaFactor_;
	}

	parse_argument(argc, argv, "-dscale", dscale_);

	loop.run();
	return 0;
}