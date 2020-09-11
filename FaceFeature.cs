using System;
using System.ComponentModel;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Media3D;
using Microsoft.Kinect;
using Microsoft.Kinect.Face;

using System.IO;
using System.Collections.Generic;
using Microsoft.Win32;
using System.Threading;

using System.Xml.Serialization;
using System.Windows.Media.Imaging;

namespace FFEA
{
    /// <summary>
    /// This a data strcture that represents a face in a frame
    /// </summary>
    public class FaceFrame
    {

        public Point3D[] Frame { get; set; }

        public FaceFrame()
        {
            Frame = new Point3D[1347];
        }

        public FaceFrame(Point3D[] Points)
        {
            Frame = new Point3D[1347];
            Frame = Points;
        }

        public Point3D GetPointByIndex(int index)
        {
            return Frame[index];
        }




        private double EuclideanDistance(Point3D a, Point3D b)
        {
            return Math.Sqrt(Math.Pow((a.X - b.X), 2) + Math.Pow((a.Y - b.Y), 2) + Math.Pow((a.Z - b.Z), 2));
        }

        /// <summary>
        /// This will return an aligned faceframe from the orignal by using DistanceSigniture method
        /// </summary>
        /// <returns></returns>
        public getDsitanceToNoseTipInOneFrame ExtractToFaceFrameSignature()
        {

            getDsitanceToNoseTipInOneFrame dc = new getDsitanceToNoseTipInOneFrame();

            foreach (int m in Enum.GetValues(typeof(FeaturePoints)))
            {

                dc.Distances.Add(EuclideanDistance(Frame[m], Frame[18]));


            }
            return dc;
        }



    }

    /// <summary>
    /// This is a data structure that represent the expression template of the subject
    /// </summary>
    public class Expression
    {
        public Expression()
        {
            expression = new List<getDsitanceToNoseTipInOneFrame>();
            description = "";// expression type
        }

        public List<getDsitanceToNoseTipInOneFrame> expression { get; set; }
        public string description { get; set; }



    }



    /// <summary>
    /// This is a data structure that represents the datas collected from the user
    /// </summary>
    [XmlRoot("FaceFeature")]
    public class FaceFeature
    {


        public FaceFeature()
        {
            ID = 0;
            Ethnicity = "";
            ExpressionType = "";
            IsMale = true;
            Age = 0;
            DepthFootage = new List<FaceFrame>();
            RGBFootage = new List<byte[]>();
            IsExpression = new List<bool>();
        }

        public int ID { set; get; }
        public string Ethnicity { set; get; }
        public string ExpressionType { set; get; }
        public bool IsMale { set; get; }
        public int Age { set; get; }
        public List<FaceFrame> DepthFootage { set; get; }
        public List<byte[]> RGBFootage { set; get; }
        public List<bool> IsExpression { set; get; }



        /// <summary>
        /// Method to extract the total distance move for points to the nose tip in a footage
        /// </summary>
        public getDsitanceToNoseTipInOneFrame averageDistance()
        {
            List<getDsitanceToNoseTipInOneFrame> oneFootage = new List<getDsitanceToNoseTipInOneFrame>();

            oneFootage.Add(new getDsitanceToNoseTipInOneFrame()); //add new footage in all
            for (int i = 0; i < DepthFootage.Count; i++) //number of frames
            {
                oneFootage.Add(DepthFootage[i].ExtractToFaceFrameSignature());
            }
            getDsitanceToNoseTipInOneFrame temp = new getDsitanceToNoseTipInOneFrame();
            double res = 0;
            for (int i = 0; i < 28; i++)
            {
                for (int k = 1; k < oneFootage.Count - 1; k++)
                {
                    res = Math.Abs(oneFootage[k + 1].Distances[i] - oneFootage[k].Distances[i]);
                    res += res;
                }
                temp.Distances.Add(res* 1000);
            }
            return temp;
        }

        /// <summary>
        /// return the expression 
        /// </summary>
        public Expression expression()
        {
            List<getDsitanceToNoseTipInOneFrame> oneFootage = new List<getDsitanceToNoseTipInOneFrame>();

            oneFootage.Add(new getDsitanceToNoseTipInOneFrame()); //add new footage in all

            for (int i = 0; i < DepthFootage.Count; i++) //number of frames
            {

                oneFootage.Add(DepthFootage[i].ExtractToFaceFrameSignature());

            }

            Expression temp = new Expression();
            temp.expression = oneFootage;
            temp.description = ExpressionType;

            return temp;
        }

    }
}
