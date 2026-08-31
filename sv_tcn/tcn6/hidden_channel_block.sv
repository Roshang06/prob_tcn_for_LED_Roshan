`include "adder_tree_block.sv"
`include "Qxxmultiply.sv"

module hidden_channel_block # (parameter int NUM_TAPS, DATA_WIDTH, SKIPCONN, TEST, LAYER_NUM, HIDDEN_CH_NUM, RESAMPLE, parameter [0:55] MODEL_TYPE) (
    input logic signed [0:NUM_TAPS-1][DATA_WIDTH-1:0] input_reg,
    input clk, reset,
    output logic [DATA_WIDTH-1:0] out
);
logic signed [DATA_WIDTH-1:0] weights [NUM_TAPS];
logic signed [DATA_WIDTH-1:0] bias [0:0];
//get weights from file todo: add error handling for retrieving from a file - The order of the weights matters in this setup
initial begin
    string fileName;
    if (RESAMPLE == 2) begin
        fileName = $sformatf("TestingData/Test%0d/%s/readout_weight/channel%0d.mem", TEST, MODEL_TYPE, HIDDEN_CH_NUM);
        $readmemh(fileName, weights, 0, NUM_TAPS-1);

        fileName = $sformatf("TestingData/Test%0d/%s/readout_bias/channel%0d.mem", TEST, MODEL_TYPE, HIDDEN_CH_NUM);
        $readmemh(fileName, bias);
    end else begin
        fileName = $sformatf("TestingData/Test%0d/%s/tcn_%0d_conv_weight/channel%0d.mem", TEST, MODEL_TYPE, LAYER_NUM, HIDDEN_CH_NUM);
        $readmemh(fileName, weights, 0, NUM_TAPS-1);

        fileName = $sformatf("TestingData/Test%0d/%s/tcn_%0d_conv_bias/channel%0d.mem", TEST, MODEL_TYPE, LAYER_NUM, HIDDEN_CH_NUM);
        $readmemh(fileName, bias);
    end
end

logic signed [0:NUM_TAPS][31:0] rounded_out;
logic signed [31:0] finalSum;
always_ff @(posedge clk) begin
    if (reset) begin
        rounded_out[NUM_TAPS] <= 31'($signed(bias[0]));
    end
end

generate
    for (genvar i = 0; i < NUM_TAPS; i++) begin: multiply_inst
        Qxxmultiply #(.DATA_WIDTH(DATA_WIDTH)) multiply_block (
            .clk(clk), 
            .reset(reset), 
            .input_reg(input_reg[i]), 
            .weight(weights[i]), 
            .final_product(rounded_out[i])
        );
    end
endgenerate

adder_tree_block # (.NUM(NUM_TAPS+1), .DATA_WIDTH(32)) adder_tree (
    .clk(clk), 
    .reset(reset), 
    .nums(rounded_out), 
    .sum(finalSum)
);

generate
    string fileName;
    if (RESAMPLE == 1) begin: resample_weight_generate //generates this for the very first layer only
        logic signed [DATA_WIDTH-1:0] resample_weight [0:0];
        logic signed [DATA_WIDTH-1:0] resample_bias [0:0];
        //logic signed [DATA_WIDTH-1:0] resampled_input;

        initial begin
            fileName = $sformatf("TestingData/Test%0d/%s/tcn_%0d_resample_weight/channel%0d.mem", TEST, MODEL_TYPE, LAYER_NUM, HIDDEN_CH_NUM);
            $readmemh(fileName, resample_weight);
            fileName = $sformatf("TestingData/Test%0d/%s/tcn_%0d_resample_bias/channel%0d.mem", TEST, MODEL_TYPE, LAYER_NUM, HIDDEN_CH_NUM);
            $readmemh(fileName, resample_bias);
        end

        // assign resampled_input = Q88clip(Q88multiply(input_reg[NUM_TAPS-1], resample_weight[0]) + resample_bias[0]);
        // assign out = (accumulator >= 0) ?  Q88clip(accumulator + resampled_input): (resampled_input); //relu and resampled input

        int signed pre_bias;
        Qxxmultiply # (.DATA_WIDTH(DATA_WIDTH)) resample_multiply_block (
            .clk(clk), 
            .reset(reset), 
            .input_reg(input_reg[NUM_TAPS-1]), 
            .weight(resample_weight[0]), 
            .final_product(pre_bias)
        );

        int signed unclipped_resampled_input;
        int signed preclipped;
        int signed relu_applied;

        localparam int delay = 2; // Formula for synchronization delay: delay = roundUpToInt(log_2(Kernel_size + 1)) - 1
        logic signed [0:delay][DATA_WIDTH-1:0] waiting_area; 
        always_ff @(posedge clk) begin
            if (reset) begin
                unclipped_resampled_input <= 0;
                preclipped <= 0;
                relu_applied <= 0;
            end else begin
                unclipped_resampled_input <= pre_bias + resample_bias[0];
                waiting_area[0] <= Q88clip(unclipped_resampled_input);

                for (int i = 1; i <= delay; i++) begin
                    waiting_area[i] <= waiting_area[i-1];
                end

                relu_applied <= (finalSum > 0) ? finalSum: 0;
                preclipped <= relu_applied + waiting_area[delay];
                out <= Q88clip(preclipped);
            end
        end
    end else if (RESAMPLE == 2) begin
        //assign out = Q88clip(accumulator); //no relu or skip connection for the readout
        always_ff @(posedge clk) begin
            if (!reset) out <= Q88clip(finalSum);
        end
    end else begin // Below is the default generation for all other layers
        //assign out = (accumulator >= 0) ?  Q88clip(accumulator + input_reg[SKIPCONN]): input_reg[SKIPCONN]; //relu and residual input added in
        int signed preclipped;
        int signed relu_applied;
        always_ff @(posedge clk) begin
            if (reset) begin
                preclipped <= 0;
                relu_applied <= 0;
            end else begin
                relu_applied <= (finalSum > 0) ? finalSum: 0;
                preclipped <= relu_applied + input_reg[SKIPCONN];
                out <= Q88clip(preclipped);
            end
        end
    end
endgenerate

function automatic int Q88clip(int signed a);
    localparam int maximumVal = 2 ** (DATA_WIDTH - 1) - 1;
    localparam int minimumVal = -1 * (2 ** (DATA_WIDTH - 1));
    if (a > maximumVal) return maximumVal;
    else if (a < minimumVal) return minimumVal;
    else return a;
endfunction
endmodule

/*
localparam qBitShift = DATA_WIDTH/2;
logic signed [0:NUM_TAPS][(DATA_WIDTH*2)-1:0] product; //bias is the last element, so this contains NUM_TAPS + 1 elements
logic signed [0:NUM_TAPS][(DATA_WIDTH*2)-1:0] product_out;
logic signed [0:NUM_TAPS][(DATA_WIDTH*2)-1:0] rounded;
logic signed [0:NUM_TAPS][(DATA_WIDTH*2)-1:0] rounded_out;
logic signed [0:NUM_TAPS-1][DATA_WIDTH-1:0] a;
logic signed [0:NUM_TAPS-1][DATA_WIDTH-1:0] b;
int signed finalSum;
adder_tree_block # (.NUM(NUM_TAPS+1), .DATA_WIDTH(DATA_WIDTH*2)) adder_tree (.clk(clk), .reset(reset), .nums(rounded_out), .sum(finalSum));
always_ff @(posedge clk) begin
    if (reset) begin
        for (int i = 0; i < NUM_TAPS; i++) begin
            product[i] <= '0;
        end
        product[NUM_TAPS] <= (DATA_WIDTH*2)'(signed'(bias));
        a <= '0;
        b <= '0;
        product_out <= '0;
        rounded <= '0;
        rounded_out <= '0;
        finalSum <= '0;
    end else begin
        for (int i = 0; i < NUM_TAPS; i++) begin
            a[i] <= input_reg[i];
            b[i] <= weights[i];
            product[i] <= a[i] * b[i];
            product_out[i] <= product[i];
            rounded[i] <= (product_out[i] + (1 <<< (qBitShift-1))) >>> qBitShift; //todo: bias isnt sent into adder tree
            rounded_out[i] <= rounded[i];
        end
        // $display("Layer %0d Channel %0d", LAYER_NUM, HIDDEN_CH_NUM);
        // $display("      in_mux: %0d %f %h, weight_mux: %0d %f %h, Prod: %0d %f %h, Acc: %0d %f %h", 
        //     in_mux,              real'(in_mux) / 256.0,          in_mux, 
        //     weight_mux,          real'(weight_mux) / 256.0,      weight_mux, 
        //     product,                      real'(product) / 256.0,                  product, 
        //     accumulator,                  real'(accumulator) / 256.0,              accumulator
        //     );
        
    end
end
*/

// helper functions
// function automatic int Q88multiply(int a, int b);
//     localparam int qBitShift = DATA_WIDTH/2;
//     int prod;
//     prod = (a * b + (1 <<< (qBitShift-1))) >>> qBitShift;
//     return prod;
// endfunction


