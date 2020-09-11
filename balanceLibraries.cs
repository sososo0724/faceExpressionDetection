using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Microsoft.Samples.Kinect.HDFaceBasics
{
    public class balanceLibraries
    {

        /// <summary>
        /// rearrange the database for fair and balanced k-fold cross-validation
        /// </summary>
        /// <param name="expression_list"></param>
        public List<string> balanceLibrary(List<string> expression)
        {
            Random random = new Random();


            List<string> storeChosenExpressionDataBase = new List<string>();

            List<string> wantedFiles = new List<string>();
            foreach (string file in Directory.EnumerateFiles(@"FFELibrary", "*.fp"))
            {
                wantedFiles.Add(file);
            }
            foreach (var file in wantedFiles)
            {
                for (int i = 0; i < expression.Count(); i++)
                {
                    if (file.Contains(expression[i]))
                    {
                        storeChosenExpressionDataBase.Add(file); //retrive the wanted expression in database
                    }
                }

            }

            for (int i = 0; i < wantedFiles.Count(); i++)
            {
                for (int k = 0; k < storeChosenExpressionDataBase.Count(); k++)
                {
                    if (wantedFiles[i] == storeChosenExpressionDataBase[k])
                    {
                        wantedFiles.RemoveAt(i);  //remove the certain expressions in database
                    }
                }
            }

            List<string> createMore = new List<string>();

            for (int i = 0; i < wantedFiles.Count() - 2; i++)
            {
                createMore.Add(wantedFiles[i]);
                createMore.Add(wantedFiles[i + 1]);
                createMore.Add(wantedFiles[i + 2]);
            }

            int index = random.Next(wantedFiles.Count);
            List<string> storeSortedData = new List<string>();

            storeSortedData = createMore.OrderBy(x => index).Take(storeChosenExpressionDataBase.Count()).ToList();//randomly take number of chosen expression count


            List<string> balancedDatabase = new List<string>();

            for (int k = 0; k < storeSortedData.Count(); k++) //add both data into final list
            {
                balancedDatabase.Add(storeSortedData[k]);
                balancedDatabase.Add(storeChosenExpressionDataBase[k]);
            }

            return balancedDatabase;
        }

    }

}
