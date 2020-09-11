using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Xml;
using System.Xml.Serialization;

namespace Microsoft.Samples.Kinect.HDFaceBasics
{
    /// <summary>
    /// Interaction logic for FeatureExeractionForAll.xaml
    /// </summary>
    public partial class Statistic : Window
    {


        public Statistic(Dictionary<string, int> SatisticalData)
        {
            InitializeComponent();

            List<TextBlock> Titles = new List<TextBlock>();
            List<TextBlock> Numbers = new List<TextBlock>();
            List<Rectangle> Rectabgles = new List<Rectangle>();

            int count = 0;
            int total = 0;
            int witdth = 400 / SatisticalData.Count - 10;

            //count how many records in total
            foreach (string str in SatisticalData.Keys)
            {
                total += SatisticalData[str];
            }

            foreach (string str in SatisticalData.Keys)
            {
                //plot type text
                Titles.Add(new TextBlock());
                Titles[count].Text = str;
                Canvas.SetBottom(Titles[count], 25);
                Canvas.SetLeft(Titles[count], 50 + count * 8 + count * witdth);
                MyCanvas.Children.Add(Titles[count]);

                //plot bar chart
                Rectabgles.Add(new Rectangle());
                Rectabgles[count].Height = 300 * SatisticalData[str] / total;
                Rectabgles[count].Width = 200 / SatisticalData.Count;
                Rectabgles[count].Fill = new SolidColorBrush(Color.FromRgb((byte)128, (byte)(255 / (count + 1)), (byte)(255 / (count + 1))));
                Canvas.SetBottom(Rectabgles[count], 50);
                Canvas.SetLeft(Rectabgles[count], 35 + count * 10 + count * witdth);
                MyCanvas.Children.Add(Rectabgles[count]);

                //plot numbers
                Numbers.Add(new TextBlock());
                Numbers[count].Text = SatisticalData[str].ToString();
                Canvas.SetBottom(Numbers[count], 55 + Rectabgles[count].Height);
                Canvas.SetLeft(Numbers[count], 50 + count * 10 + count * witdth);
                MyCanvas.Children.Add(Numbers[count]);

                count++;

            }

        }
    }
}