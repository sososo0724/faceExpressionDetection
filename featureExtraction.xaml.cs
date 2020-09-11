using FFEA;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Globalization;
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
using System.Xml.Serialization;

namespace Microsoft.Samples.Kinect.HDFaceBasics
{
    /// <summary>
    /// Interaction logic for featureExtraction.xaml
    /// </summary>
    public partial class featureExtraction : Window
    {

        public List<List<double>> getForOneType { get; set; } //store 2d array of a footage
        public List<int> storeFeaturePoint = new List<int>();

        public Dictionary<int, List<double>> getOneFootage { get; set; }

        public featureExtraction()
        {
            InitializeComponent();

            this.DataContext = this;

            foreach (int m in Enum.GetValues(typeof(FeaturePoints)))
            {
                storeFeaturePoint.Add(m);

            }

            getOneFootage = new Dictionary<int, List<double>>();

            List<double> sample = new List<double>();
            for (int i = 0; i < storeFeaturePoint.Count; i++)
            {
                sample.Add(i);
            }
            getOneFootage.Add(0, sample);


            int count = 0;

            foreach (List<double> lst in getOneFootage.Values)
            {
                if (lst.Count > count)
                {
                    for (int k = count; k < lst.Count; k++)
                    {
                        DataGridTextColumn col = new DataGridTextColumn();
                        col.Header = storeFeaturePoint[k];
                        col.Binding = new Binding(string.Format("Value[{0}]", k));
                        dataGrid.Columns.Add(col);
                    }
                    count = lst.Count;
                }
            }

            this.dataGrid.ItemsSource = getOneFootage;
        }



        /// <summary>
        /// button click event to get the information of loaded footage
        /// </summary>
        /// <param name="file"></param>
        /// <param name="e"></param>
        private void totalDistance_clicked(object sender, RoutedEventArgs e)
        {
            {

                Dictionary<int, double> getTotal = new Dictionary<int, double>();

                int k;

                double res = 0;
                for (k = 0; k < storeFeaturePoint.Count; k++)
                {
                    for (int i = 1; i < getOneFootage.Count - 1; i++)
                    {
                        res = Math.Abs(getOneFootage[i + 1][k] - getOneFootage[i][k]);
                        res = +res;
                    }
                    getTotal.Add(storeFeaturePoint[k], Math.Round(res * 2000, 9, MidpointRounding.AwayFromZero));
                }
                dataGrid1.ItemsSource = getTotal;
            }
        }


        //function to get the one footage's average distance movement acorss all frame
        public List<List<double>> decodeFootage(string file)
        {
            getForOneType = new List<List<double>>();
            List<double> tempAll = new List<double>();
            FFEA.Expression getDistance = new FFEA.Expression();
            XmlSerializer deserializer = new XmlSerializer(typeof(FaceFeature));
            TextReader reader = new StreamReader(file);

            getDistance = ((FaceFeature)deserializer.Deserialize(reader)).expression();

            for (int i = 1; i < getDistance.expression.Count(); i++)
            {
                List<double> temp = new List<double>();

                for (int k = 0; k < Enum.GetValues(typeof(FeaturePoints)).Length; k++)
                {
                    temp.Add(Math.Round(Convert.ToDouble(getDistance.expression[i].Distances[k]), 9, MidpointRounding.AwayFromZero));
                }
                getForOneType.Add(temp);
            }
            return getForOneType;
        }

        /// <summary>
        /// get the total dispplacement of points across all frames
        /// </summary>
        /// <param name="file"></param>
        public List<double> getOne(string file)
        {
            getForOneType = new List<List<double>>();
            List<double> tempAll = new List<double>();
            FFEA.Expression getDistance = new FFEA.Expression();
            XmlSerializer deserializer = new XmlSerializer(typeof(FaceFeature));
            TextReader reader = new StreamReader(file);

            getDistance = ((FaceFeature)deserializer.Deserialize(reader)).expression();
            double res = 0;

            for (int i = 1; i < getDistance.expression.Count(); i++)
            {
                List<double> temp = new List<double>();

                for (int k = 0; k < Enum.GetValues(typeof(FeaturePoints)).Length; k++)
                {
                    temp.Add(Math.Round(Convert.ToDouble(getDistance.expression[i].Distances[k]), 9, MidpointRounding.AwayFromZero));
                }
                getForOneType.Add(temp);
            }

            for (int k = 0; k < storeFeaturePoint.Count(); k++)
            {
                for (int i = 1; i < getForOneType.Count() - 1; i++)
                {
                    res = Math.Abs(getForOneType[i + 1][k] - getForOneType[i][k]);
                    res = +res;

                }
                tempAll.Add(res*1000); //get the average movement for selected across all frames
            }
            return tempAll;
        }

        private void DataGrid_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {

        }

    }
}
