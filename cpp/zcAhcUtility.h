#ifndef _ZC_AHC_UTILITY_H_
#define _ZC_AHC_UTILITY_H_

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <set>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>

#include "AHCPlaneSeg.hpp" //zc: 这里用于构造 虚拟的PlaneSeg实例

using std::vector;
using std::set;
using ahc::PlaneSeg;

using Eigen::aligned_allocator;
using Eigen::Affine3d;
using Eigen::Affine3f;
using Eigen::Matrix3f;
using Eigen::Matrix3d;
using Eigen::Matrix3Xf;
using Eigen::Vector3d;
using Eigen::Vector3f;
using Eigen::Map;

using Eigen::JacobiSVD;
using Eigen::ComputeFullU;
using Eigen::ComputeFullV;

typedef vector<Affine3d, aligned_allocator<Affine3d>> Affine3dVec;
typedef vector<Affine3f, aligned_allocator<Affine3f>> Affine3fVec;

#define PI 3.14159265358979323846

//zc: 量纲转换
const float M2MM = 1e3;
const float MM2M = 1e-3;

//立方体配准, opengl 渲染相关    //2017-1-10 23:09:04
const int WIN_WW = 640,
		  WIN_HH = 480;

class Cube;

//@brief 输出 t3+R9 到
void processPoses(const char *fn, const Affine3dVec &poses);

//@brief 输出平面参数: 法向,质心,曲率, mse(这啥用?)
void printPlaneParams(const double *normals, const double *center, double curvature, double mse);
void printPlaneParams(const PlaneSeg &planeSeg);

//@brief 点乘, 向量长度不固定
double dotProd(const double *v1, const double *v2, size_t len = 3);

//@brief 向量模长
double norm(const double *v, size_t len = 3);

//@brief 两点欧氏距离
double dist(const double *v1, const double *v2, size_t len = 3);

//@brief 叉积 (v1 x v2)=v3; 只管三维3D;
//@param[out] v3, 输出叉积结果; 需要外部预分配内存
void crossProd(const double *v1, const double *v2, double *v3);

//@brief 施密特正交化, 目前只管前两轴, 第三轴这里不管(外部直接做叉积); 默认向量也是3D的; 输入不必是单位向量
//@param[in] v1, 主参考轴
//@param[out] newv1: =v1/|v1| 单位化; 需要外部预分配内存
//@param[out] newv2: v2 参照 v1 正交化之后的输出; 需要外部预分配内存
void schmidtOrtho(const double *v1, const double *v2, double *newv1, double *newv2, size_t len = 3);

//@brief 列主元高斯消去法; 见: https://www.oschina.net/code/snippet_76_4375
//A为系数矩阵，x为解向量，若成功，返回true，否则返回false，并将x清空。
bool RGauss(const vector<vector<double> > & A, vector<double> & x);

//@brief 其实是修改 pSegMat, 并不 imshow
//@param[in] labelMat: 如 ahc.PlaneFitter.membershipImg 
//@param[in] pSegMat: 调试观察mat; 为啥用指针: 其实用传值也行, 但是不能传引用, 否则: cannot bind non-const lvalue reference of type 'int&' to an rvalue of type 'int'
//void showLabelMat(cv::Mat labelMat, cv::Mat *pSegMat = 0);
void annotateLabelMat(cv::Mat labelMat, cv::Mat *pSegMat = 0); //改名

//@brief 仿照 ahc.PlaneFitter.run 做成模板函数, 放在头文件
//@param[in] &pointsIn, 模板类, 自己实现 (e.g., NullImage3D, OrganizedImage3D)
//@param[in] &idxVecs, index vector of vector 
//@param[in] doRefine 默认=true [DEPRECATED]
template <class Image3D>
vector<PlaneSeg> zcRefinePlsegParam(const Image3D &pointsIn, const vector<vector<int>> &idxVecs/*, bool doRefine = true*/){
	vector<PlaneSeg> plvec; //存放各个平面参数

	size_t plCnt = idxVecs.size();
	for(size_t i=0; i<plCnt; i++){
		const vector<int> &idxs = idxVecs[i];
		PlaneSeg tmpSeg(pointsIn, idxs);
		plvec.push_back(tmpSeg);

		//if(dbg2_)
			//printPlaneParams(tmpSeg);
	}//for

	return plvec;
}//zcRefinePlsegParam

//@brief zc: 通过 lblMat, 将未确定/无方向的三轴直线, 确定为三轴射线(有方向)
//@param[in] dmap, 原深度图, 量纲mm
//@param[in] orig, 原点, 即 t3; 量纲m
//@param[in] axs, 三轴直线, 即 R9; 量纲无, 三个单位向量; 可能输入时已经是右手坐标系, 但返回值输出可能破坏此属性
vector<double> zcAxLine2ray(const cv::Mat &dmap, const vector<double> &orig, const vector<double> &axs,
	double fx, double fy, double cx, double cy);

//@brief zc: 找两两正交的三个平面, 组成一个三元组 (可能有多组, e.g., 立方体一个竖直棱的上下顶角, 或干扰物造成), 
//示意图见: http://codepad.org/he35YCTh
//@return ortho3tuples, 是pl-idx-tup3, 不是实际平面参数; 后期可能用不到了
//@param[in] plvec
//@param[in] lblMat 平面划分 label 图, 如 ahc.PlaneFitter.membershipImg 
//@param[out] cubeCandiPoses, vec-vec, 初始必须是空, N*12(t3+R9), 【del】R9不是真的旋转矩阵, 甚至不正交【del】. 改成 R9是旋转矩阵(nearest orhto 解得), 按行存储(row-major)??? 【不确定】
//@param[out] prev 【放弃】
//@param[in/out] dbgMat, e.g.: pf.run 输出的调试观察 Mat
vector<vector<int>> zcFindOrtho3tup(const vector<PlaneSeg> &plvec, const cv::Mat &lblMat,
	double fx, double fy, double cx, double cy,
	vector<vector<double>> &cubeCandiPoses, cv::OutputArray dbgMat = cv::noArray());

//@brief 输入一串 cubeCandiPoses-vec, 输出一个 优化后的 cubePose, 从 cu->cam
//@return 注意是 3d, kinfu 用的是 3f, 要正确转换
//@param[in] cubeCandiPoses, vec-vec, 每行 12列(t3+R9); 之前 zcFindOrtho3tup 改过策略, 使 R9 必然是旋转矩阵; 其实最多 size()==2, 即单视角俩顶角
Affine3d getCuPoseFromCandiPoses(const vector<vector<double>> &cubeCandiPoses);

//@brief zc: 根据内参, 将 3d相机坐标转为 2d像素; 返回int型
cv::Point getPxFrom3d(const Vector3d &pt3d, float fx, float fy, float cx, float cy);
//@brief 返回 float 型
cv::Point2f getPx2fFrom3d(const Vector3d &pt3d, float fx, float fy, float cx, float cy);

//@brief zc: 根据 crnrTR, dmap 求解实际相机棱边线段(非直线), 即 1+3 个顶点
//@param[in] crnrTR, t3+R9=12列, 取自 cubeCandiPoses; t部分量纲: 米
//@param[in] cuSideVec, 立方体三边长度, 量纲: 米
//@param[in] dmap, 深度图, ushort
//@param[in] labelMat, 如 ahc.PlaneFitter.membershipImg, 
//@param[out] pts4, 输出值: 1*12 vec, 4个坐标点
//@return 若找到4点:true; 若因某些原因中断:false
//vector<double> getCu4Pts(const vector<double> &crnrTR, const vector<float> &cuSideVec,const cv::Mat &dmap, cv::Mat &labelMat, float fx, float fy, float cx, float cy);
bool getCu4Pts(const vector<double> &crnrTR, const vector<float> &cuSideVec,const cv::Mat &dmap, cv::Mat &labelMat, float fx, float fy, float cx, float cy, vector<double> &pts4);

//@brief zc: 手动渲染立方体深度图, 此前尝试了 opengl(pcl控制, fbo也弄不好, 不会用), vtk(无法使用真实内参), 此函数假设视点在原点 000, 故不传相机外参做参数
//2017-1-14 23:34:23
//@param[in] Cube, 相机坐标系下, 立方体模型, 
//@param[in] intr=(fx,fy,cx,cy), 相机内参
//@return 返回 16uc1 的深度图
cv::Mat zcRenderCubeDmap(const Cube &cube, float fx, float fy, float cx, float cy);

//@brief zc: 判断直线与平面是否相交; 参考: https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
//@param[in] (L0, L), 直线点斜式
//@param[in] (plnorm, plp0), 平面法向与某点(如某面片质心)
//@param[out] ptInters, 直线与平面交点
bool isLinePlaneIntersect(const Vector3d &L0, const Vector3d &L, const Vector3d &plnorm, const Vector3d &plp0, Vector3d &ptInters);

//@brief 绘制虚线, 目前仅用于 Cube.drawContour 消隐轮廓绘制; 形参列表仿照 cv::line
//代码参考: http://answers.opencv.org/question/10902/how-to-plot-dotted-line-using-opencv/
void zcDashLine(CV_IN_OUT cv::Mat& img, cv::Point pt1, cv::Point pt2, const cv::Scalar& color/*, int thickness=1, int lineType=8, int shift=0*/);

//@brief zc: 立方体面片类
class Facet{
	//TODO  //2017-1-14 23:27:07
public:
	vector<int> vertIds_;
	Vector3d center_, normal_;

	//@brief 判定面片是否含有某顶点 id
	bool isContainVert(const int vertId) const{
		const size_t vCnt = vertIds_.size();
		for(size_t i=0; i<vCnt; i++){
			if(vertId == vertIds_[i])
				return true;
		}

		return false;
	}//isContainVert
};
class Cube{
private:
	//vector<Vector3d> cuVerts8_;
	//vector<Facet>

	//@brief 每个顶点被三邻面共有, 此 vec-vec 记录顶点 [i] 的三邻面 id, 8*3; 此变量目前仅用于 drawContour
	//[DEPRECATED] 对于面片是否包含顶点问题, 改用 isContainVert 判定 //2017-1-17 23:28:50
	vector<set<int>> vertAdjFacetIds_;

public:
	vector<Vector3d> cuVerts8_; //1x8
	//void setVerts8(const vector<Vector3d> &cuVerts8){ cuVerts8_ = cuVerts8; }
	//vector<int*> faceVertIdVec_; //6x4
	vector<Facet> facetVec_;
	vector<vector<int>> edgeIds_;

	Cube(){}
	Cube(const Cube &cuOther, const Affine3d &affine);

	//@brief 根据顶点序号 vertIds, 添加 facet
	//@param[in] vertIds, 顶点序号 0~7 的某4个
	void addFacet(const vector<int> &vertIds);
	void addEdgeId(vector<int> &edge);

	bool isLineFacetIntersect(const Vector3d &L0, const Vector3d &L, const Facet &facet, Vector3d &ptInters) const;
	bool isLineFacetIntersect(const Vector3d &L0, const Vector3d &L, int faceId, Vector3d &ptInters) const{
		return isLineFacetIntersect(L0, L, facetVec_[faceId], ptInters);
	}
	//@brief 视线(过原点与顶点L), 是否与面片 faceId 相交
	//@param[in] L, 是顶点, 但因起点在原点, 所以也同时代表原点发出的射线
	bool isVrayFacetIntersect(const Vector3d &L, int faceId, Vector3d &ptInters) const{
		Vector3d L0(0,0,0);
		return isLineFacetIntersect(L0, L, facetVec_[faceId], ptInters);
	}//isVrayFacetIntersect

	//@brief 不输出线面交点版本, 无 "Vector3d &ptInters" 参数
	bool isVrayFacetIntersect(const Vector3d &L, int faceId) const{
		Vector3d ptInters_tmp; //占位符, 并不输出
		Vector3d L0(0,0,0);
		return isLineFacetIntersect(L0, L, facetVec_[faceId], ptInters_tmp);
	}//isVrayFacetIntersect


	//@brief 判断是否已经正常初始化填充过
	bool isValid(){ return this->cuVerts8_.size() != 0; }

	//@brief 绘制到 2D cv::mat
	//@param[out], dstCanvas, 被绘制画布, 若无内容, 黑背景; 若已有内容, 叠加绘制
	//@param[in], (fx,fy, cx,cy) 相机内参
    //@param[in], color, 待绘制线框轮廓颜色
	//@param[in], hideLines, 绘制时是否消隐
	void drawContour(cv::Mat dstCanvas, double fx, double fy, double cx, double cy, const cv::Scalar& color, bool hideLines = false);
};


#endif //_ZC_AHC_UTILITY_H_
