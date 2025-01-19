pub struct Model {
    pub inputNodes:usize,// = 0;
    pub hiddenNodes:usize,// = 0;
    pub outputNodes:usize,// = 0;
    pub epochs:usize,// = 1;
    pub learningRate:float,//  = 0.3;
    pub scalingFactor:float,// = 1.0;
    pub shuffleData:bool,// = true;
    pub validationSplit:float, //= 0.1;
    pub dataRows:usize,// = 0;
    pub splitIndex:usize,// = 0;
}