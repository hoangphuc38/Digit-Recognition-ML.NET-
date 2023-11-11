using DemoML.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.ML;
using System.Diagnostics;
using System.Drawing;
using System.Text;
using System;

namespace DemoML.Controllers
{
    public class HomeController : Controller
    {
        private readonly ILogger<HomeController> _logger;

        //private readonly PredictionEnginePool<ModelInput, ModelOutput> _predictionEnginePool;
        private MLModel model;
        private const int SizeOfImage = 32;
        private const int SizeOfArea = 4;

        public HomeController(ILogger<HomeController> logger)
        {
            _logger = logger;
            //_predictionEnginePool = predictionEnginePool;
        }

        public IActionResult Index()
        {
            return View();
        }

        public IActionResult Privacy()
        {
            return View();
        }

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }

        [HttpPost]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        public IActionResult Upload(string base64Image)
        {
            if (string.IsNullOrEmpty(base64Image))
            {
                return BadRequest(new { prediction = '-', dataset = string.Empty });
            }
            string replaceImage = base64Image.Replace("data:image/png;base64,", String.Empty);
            var pixelValues = GetPixelValuesFromImage(replaceImage);

            float[] myArray = pixelValues.ToArray();
            int i = 0;
            foreach (float number in pixelValues)
            {
                myArray[i] = number;
                i++;
            }

            var input = new MLModel.ModelInput
            {
                Col0 = myArray[0],
                Col1 = myArray[1],
                Col2 = myArray[2],
                Col3 = myArray[3],
                Col4 = myArray[4],
                Col5 = myArray[5],
                Col6 = myArray[6],
                Col7 = myArray[7],
                Col8 = myArray[8],
                Col9 = myArray[9],
                Col10 = myArray[10],
                Col11 = myArray[11],
                Col12 = myArray[12],
                Col13 = myArray[13],
                Col14 = myArray[14],
                Col15 = myArray[15],
                Col16 = myArray[16],
                Col17 = myArray[17],
                Col18 = myArray[18],
                Col19 = myArray[19],
                Col20 = myArray[20],
                Col21 = myArray[21],
                Col22 = myArray[22],
                Col23 = myArray[23],
                Col24 = myArray[24],
                Col25 = myArray[25],
                Col26 = myArray[26],
                Col27 = myArray[27],
                Col28 = myArray[28],
                Col29 = myArray[29],
                Col30 = myArray[30],
                Col31 = myArray[31],
                Col32 = myArray[32],
                Col33 = myArray[33],
                Col34 = myArray[34],
                Col35 = myArray[35],
                Col36 = myArray[36],
                Col37 = myArray[37],
                Col38 = myArray[38],
                Col39 = myArray[39],
                Col40 = myArray[40],
                Col41 = myArray[41],
                Col42 = myArray[42],
                Col43 = myArray[43],
                Col44 = myArray[44],
                Col45 = myArray[45],
                Col46 = myArray[46],
                Col47 = myArray[47],
                Col48 = myArray[48],
                Col49 = myArray[49],
                Col50 = myArray[50],
                Col51 = myArray[51],
                Col52 = myArray[52],
                Col53 = myArray[53],
                Col54 = myArray[54],
                Col55 = myArray[55],
                Col56 = myArray[56],
                Col57 = myArray[57],
                Col58 = myArray[58],
                Col59 = myArray[59],
                Col60 = myArray[60],
                Col61 = myArray[61],
                Col62 = myArray[62],
                Col63 = myArray[63],
            };
            var result = MLModel.Predict(input);
            _logger.LogInformation($"Number {result.Prediction} is returned.");
            return Ok(new
            {
                prediction = result.Prediction,
                pixelValues = string.Join(",", pixelValues)
            });
        }

        private static List<float> GetPixelValuesFromImage(string base64Image)
        {
            byte[] imageBytes = Convert.FromBase64String(base64Image);

            // resize the original image and save it as bitmap
            var bitmap = new Bitmap(SizeOfImage, SizeOfImage);
            using (var g = Graphics.FromImage(bitmap))
            {
                g.Clear(Color.White);
                using var stream = new MemoryStream(imageBytes);
                var png = Image.FromStream(stream);
                g.DrawImage(png, 0, 0, SizeOfImage, SizeOfImage);
            }

            // aggregate pixels in 4X4 area --> 'result' is a list of 64 integers
            var result = new List<float>();
            for (var i = 0; i < SizeOfImage; i += SizeOfArea)
            {
                for (var j = 0; j < SizeOfImage; j += SizeOfArea)
                {
                    var sum = 0;        // 'sum' is in the range of [0,16].
                    for (var k = i; k < i + SizeOfArea; k++)
                    {
                        for (var l = j; l < j + SizeOfArea; l++)
                        {
                            if (bitmap.GetPixel(l, k).Name != "ffffffff") sum++;
                        }
                    }
                    result.Add(sum);
                }
            }

            return result;
        }   
    }
}