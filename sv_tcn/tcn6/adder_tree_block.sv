module adder_tree_block # (parameter int NUM, DATA_WIDTH, DEPTH=-1) (
    input clk, reset,
    input logic signed [0:NUM-1][DATA_WIDTH-1:0] nums,
    output logic signed [31:0] sum
);
parameter int depth = (DEPTH >= 0) ? DEPTH: adder_tree_block_latency(NUM);

if (NUM == 1) begin: base_case1
    always_ff @(posedge clk) begin
        if (reset) sum <= '0;
        else sum <= 32'($signed(nums[0]));
    end
end else if (NUM == 2) begin: base_case2
    always_ff @(posedge clk) begin
        if (reset) sum <= '0;
        else sum <= 32'($signed(nums[0])) + 32'($signed(nums[1]));
    end
end else if (NUM == 3 && depth == 1) begin: base_case3
    always_ff @(posedge clk) begin
        if (reset) sum <= '0;
        else sum <= 32'($signed(nums[0])) + 32'($signed(nums[1])) + 32'($signed(nums[2]));
    end
end else begin: recursion
    localparam int halfnum = NUM >> 1;
    // localparam bit mismatched_depth = (($countones(NUM) == 2) && NUM[0]) && (NUM != 3);
    logic signed [31:0] sum1;
    logic signed [31:0] sum2;
    adder_tree_block # (.NUM(halfnum), .DATA_WIDTH(DATA_WIDTH), .DEPTH(depth - 1)) 
    subAdder1 (
        .clk(clk), 
        .reset(reset), 
        .nums(nums[0:halfnum-1]), 
        .sum(sum1)
    );
    adder_tree_block # (.NUM(NUM-halfnum), .DATA_WIDTH(DATA_WIDTH), .DEPTH(depth - 1)) 
    subAdder2 (
        .clk(clk), 
        .reset(reset), 
        .nums(nums[halfnum:NUM-1]), 
        .sum(sum2)
    );

    always_ff @(posedge clk) begin
        if (reset) sum <= '0;
        else sum <= sum1 + sum2;
    end
end
endmodule

function automatic int adder_tree_block_latency(int num);
    int latency = (num <= 2) ? 1: $floor(($ln((real'(num) - 1.0) / 3.0) / $ln(2.0)) + 2);
    return latency;
endfunction