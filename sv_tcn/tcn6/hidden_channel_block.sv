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
        rounded_out[NUM_TAPS] <= 32'($signed(bias[0]));
    end
    // if (LAYER_NUM == 0 && HIDDEN_CH_NUM == 0) begin
    //     $display("Layer %0d Channel %0d", LAYER_NUM, HIDDEN_CH_NUM);
    //     foreach (input_reg[i]) begin  
    //         $display("      in_mux: %0d %f %h, weight_mux: %0d %f %h, Prod: %0d %f %h, Acc: %0d %f %h", 
    //             input_reg[i],        real'(input_reg[i]) / 256.0,    input_reg[i], 
    //             weights[i],          real'(weights[i]) / 256.0,      weights[i], 
    //             rounded_out[i],      real'(rounded_out[i]) / 256.0,  rounded_out[i], 
    //             finalSum,            real'(finalSum) / 256.0,        finalSum
    //             );
    //     end
    // end
end

generate
    for (genvar i = 0; i < NUM_TAPS; i++) begin: multiply_inst
        Qxxmultiply #(.DATA_WIDTH(DATA_WIDTH)) 
        multiply_block (
            .clk(clk), 
            .reset(reset), 
            .input_reg(input_reg[i]), 
            .weight(weights[i]), 
            .final_product(rounded_out[i])
        );
    end
endgenerate

adder_tree_block # (.NUM(NUM_TAPS+1), .DATA_WIDTH(32)) 
adder_tree (
    .clk(clk), 
    .reset(reset), 
    .nums(rounded_out), //6
    .sum(finalSum) //6 + 6
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
        Qxxmultiply # (.DATA_WIDTH(DATA_WIDTH)) 
        resample_multiply_block (
            .clk(clk), 
            .reset(reset), 
            .input_reg(input_reg[NUM_TAPS-1]), 
            .weight(resample_weight[0]), 
            .final_product(pre_bias)
        );

        int signed unclipped_resampled_input;
        int signed preclipped;
        int signed relu_applied;

        localparam int delay = 2*$clog2(NUM_TAPS + 1) - 1; // Formula for synchronization delay (due to adder tree)
        logic signed [0:delay][DATA_WIDTH-1:0] waiting_line; 
        always_ff @(posedge clk) begin
            if (reset) begin
                unclipped_resampled_input <= 0;
                preclipped <= 0;
                relu_applied <= 0;
            end else begin
                unclipped_resampled_input <= pre_bias + resample_bias[0];
                waiting_line[0] <= Q88clip(unclipped_resampled_input);

                for (int i = 1; i <= delay; i++) begin
                    waiting_line[i] <= waiting_line[i-1];
                end

                relu_applied <= (finalSum > 0) ? finalSum: 0;
                preclipped <= relu_applied + waiting_line[delay];
                out <= Q88clip(preclipped); //out 12
            end
        end
    end else if (RESAMPLE == 2) begin: readout
        //assign out = Q88clip(accumulator); //no relu or skip connection for the readout
        always_ff @(posedge clk) begin
            if (!reset) out <= Q88clip(finalSum); // out 10
        end
    end else begin: middle_layer // Below is the default generation for all other layers
        //assign out = (accumulator >= 0) ?  Q88clip(accumulator + input_reg[SKIPCONN]): input_reg[SKIPCONN]; //relu and residual input added in
        localparam int delay = 2*$clog2(NUM_TAPS + 1) + 5; // Formula for synchronization delay (due to adder tree and Qmutliply)
        logic signed [0:delay][DATA_WIDTH-1:0] waiting_line;
        
        int signed preclipped;
        int signed relu_applied;
        always_ff @(posedge clk) begin
            if (reset) begin
                preclipped <= 0;
                relu_applied <= 0;
            end else begin
                waiting_line[0] <= input_reg[SKIPCONN];

                for (int i = 1; i <= delay; i++) begin
                    waiting_line[i] <= waiting_line[i-1];
                end

                relu_applied <= (finalSum > 0) ? finalSum: 0;
                preclipped <= relu_applied + waiting_line[delay];
                out <= Q88clip(preclipped); // out 12
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