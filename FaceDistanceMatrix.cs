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
    /// <sumary>
    /// get all features point distance to the nose tip in all frames
    /// </sumary>
    public class getOneExpressionDistance
    {
        public getOneExpressionDistance()
        {
          allFeatureDistance = new List<getDsitanceToNoseTipInOneFrame>();
        }

        public List<getDsitanceToNoseTipInOneFrame> allFeatureDistance { get; set; }
    }

    /// <summary>
    /// get all features point distance to the nose tip in one frame
    /// </summary>
   public class getDsitanceToNoseTipInOneFrame
    {
        public getDsitanceToNoseTipInOneFrame()
        {
            Distances = new List<double>();
        }

        public List<double> Distances { get; set; }
    }
}
