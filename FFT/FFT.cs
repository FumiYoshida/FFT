using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FFT
{
    class FFT
    {
        public double[][] Convolution(double[][] x, double[][] y)
        {
            int len = x[0].Length;
            double[][] output = new double[2][] { new double[len], new double[len] };
            for (int i = 0; i < len; i++)
            {
                double tempr = 0;
                double tempi = 0;
                for (int j = 0; j < len; j++)
                {
                    tempr += x[0][j] * y[0][len - j] - x[1][j] * y[1][len - j];
                    tempi += x[0][j] * y[1][len - j] + x[1][j] * y[0][len - j];
                }
                output[0][i] = tempr;
                output[1][i] = tempi;
            }
            return output;
        }

        public double[][] FastConvolution(double[][] x, double[][] y)
        {
            double[][] x2 = FastFourierTransform(x);
            double[][] y2 = FastFourierTransform(y);
            double[][] xy = HadamardProduct(x2, y2);
            double[][] output = InverseFastFourierTransform(xy);
            return output;
        }


        public double[][] FastFourierTransform(double[][] input)
        {
            // inputr, inputi には4^nの長さの配列が入る。
            double[] inputr = input[0];
            double[] inputi = input[1];
            int deg = (int)Math.Floor(Math.Log(inputr.Length + 0.5) / Math.Log(4));
            int ndeg = 1 << (deg * 2);
            for (int i = 1; i <= deg; i++)
            {
                int pdeg = 1 << ((deg - i) * 2);
                for (int j0 = 0; j0 < (1 << ((i - 1) * 2)); j0++)
                {
                    for (int j1 = 0; j1 < pdeg; j1++)
                    {
                        int j = j1 + j0 * pdeg * 4;
                        // バタフライ演算
                        double w1 = inputr[j] + inputr[j + pdeg] + inputr[j + 2 * pdeg] + inputr[j + 3 * pdeg];
                        double w2 = inputi[j] + inputi[j + pdeg] + inputi[j + 2 * pdeg] + inputi[j + 3 * pdeg];
                        double w3 = inputr[j] + inputi[j + pdeg] - inputr[j + 2 * pdeg] - inputi[j + 3 * pdeg];
                        double w4 = inputi[j] - inputr[j + pdeg] - inputi[j + 2 * pdeg] + inputr[j + 3 * pdeg];
                        double w5 = inputr[j] - inputr[j + pdeg] + inputr[j + 2 * pdeg] - inputr[j + 3 * pdeg];
                        double w6 = inputi[j] - inputi[j + pdeg] + inputi[j + 2 * pdeg] - inputi[j + 3 * pdeg];
                        double w7 = inputr[j] - inputi[j + pdeg] - inputr[j + 2 * pdeg] + inputi[j + 3 * pdeg];
                        double w8 = inputi[j] + inputr[j + pdeg] - inputi[j + 2 * pdeg] - inputr[j + 3 * pdeg];
                        inputr[j] = w1;
                        inputi[j] = w2;
                        inputr[j + pdeg] = w3;
                        inputi[j + pdeg] = w4;
                        inputr[j + 2 * pdeg] = w5;
                        inputi[j + 2 * pdeg] = w6;
                        inputr[j + 3 * pdeg] = w7;
                        inputi[j + 3 * pdeg] = w8;

                        // 回転因子
                        for (int k = 0; k < 4; k++)
                        {
                            w1 = Math.Cos(2 * Math.PI * j * k / pdeg / 4);
                            w2 = Math.Sin(2 * Math.PI * j * k / pdeg / 4);
                            w3 = inputr[j + k * pdeg] * w1 - inputi[j + k * pdeg] * w2;
                            w4 = inputr[j + k * pdeg] * w2 + inputi[j + k * pdeg] * w1;
                            inputr[j + k * pdeg] = w3;
                            inputi[j + k * pdeg] = w4;
                        }
                    }
                }
            }
            double[] outputr = new double[ndeg];
            double[] outputi = new double[ndeg];
            // ビット反転
            for (int i = 0; i < ndeg; i++)
            {
                int k = i;
                int k1 = 0;
                for (int j = 1; j <= deg; j++)
                {
                    k1 += (k % 4) * (1 << ((deg - j) * 2));
                    k = k / 4;
                }
                outputr[i] = inputr[k1];
                outputi[i] = inputi[k1];
            }
            return new double[2][] { outputr, outputi };
        }

        public double[][] InverseFastFourierTransform(double[][] input)
        {
            double[][] inputbar = new double[2][] { new double[input[0].Length], new double[input[0].Length] };
            for (int i = 0; i < input[0].Length; i++)
            {
                inputbar[0][i] = input[0][i];
                inputbar[1][i] = -input[1][i];
            }
            double[][] output = FastFourierTransform(inputbar);
            for (int i = 0; i < input[0].Length; i++)
            {
                output[0][i] = output[0][i];
                output[1][i] = -output[1][i];
            }
            return output;

        }

        public double[][] HadamardProduct(double[][] x, double[][] y)
        {
            int len = Math.Min(x[0].Length, y[0].Length);
            double[][] output = new double[2][] { new double[len], new double[len]};
            for (int i = 0; i < len; i++)
            {
                output[0][i] = x[0][i] * y[0][i] - x[1][i] * y[1][i];
                output[1][i] = x[0][i] * y[1][i] + x[1][i] * y[0][i];
            }
            return output;
        }
    }
}
