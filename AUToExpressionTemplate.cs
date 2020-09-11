using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Microsoft.Samples.Kinect.HDFaceBasics
{
    class AUToExpressionTemplate
    {
        /// <summary>
        /// store the expression-aus template string 
        /// </summary>
        
        public List<List<int>> templateString()
        {
            List<List<int>> template = new List<List<int>>();
            List<int> sad = new List<int>();

            sad.Add(15);
            sad.Add(17);
            sad.Add(1);
            sad.Add(4);
            
            List<int> shock = new List<int>();
            shock.Add(27);
            shock.Add(2);
            shock.Add(1);
            shock.Add(5);

            List<int> smile = new List<int>();
            smile.Add(12);
            smile.Add(6);                
            smile.Add(26);
                
            List<int> laugh = new List<int>();
            laugh.Add(6);
            laugh.Add(26);
            laugh.Add(2);
            laugh.Add(5);

            template.Add(smile);
            template.Add(sad);
            template.Add(shock);
            template.Add(laugh);
            return template;
        }
    }
  
}
