module adder_tree_block # (parameter int NUM, DATA_WIDTH, DEPTH=-1, MAX_ADDS=3) (
    input clk, reset,
    input logic signed [0:NUM-1][DATA_WIDTH-1:0] nums,
    output logic signed [31:0] sum
);
parameter int depth = (DEPTH >= 0) ? DEPTH: adder_tree_block_latency(NUM);
logic [31:0] total;

if (NUM < MAX_ADDS || (NUM == MAX_ADDS && depth == 1)) begin: base_case
    always_comb begin
        total = '0;
        foreach(nums[i]) begin
            total += nums[i];
        end
    end
    always_ff @(posedge clk) begin
        if (reset) sum <= '0;
        else sum <= total;
    end

// if (NUM == 1) begin: base_case1
//     always_ff @(posedge clk) begin
//         if (reset) sum <= '0;
//         else sum <= 32'($signed(nums[0]));
//     end
// end else if (NUM == 2) begin: base_case2
//     always_ff @(posedge clk) begin
//         if (reset) sum <= '0;
//         else sum <= 32'($signed(nums[0])) + 32'($signed(nums[1]));
//     end
// end else if (NUM == 3 && depth == 1) begin: base_case3
//     always_ff @(posedge clk) begin
//         if (reset) sum <= '0;
//         else sum <= 32'($signed(nums[0])) + 32'($signed(nums[1])) + 32'($signed(nums[2]));
//     end
end else begin: recursion
    localparam int halfnum = $floor((real'(NUM) / real'(MAX_ADDS)) + 0.5);
    // localparam int floored = NUM / MAX_ADDS;
    // localparam int remainder = NUM % MAX_ADDS;

    // logic signed [31:0] sum1;
    // logic signed [31:0] sum2;
    logic signed [0:MAX_ADDS-1][31:0] sub_sums;
    for (genvar i = 1; i < MAX_ADDS; i++) begin: inst_recurs
        adder_tree_block # (.NUM(halfnum), .DATA_WIDTH(DATA_WIDTH), .DEPTH(depth - 1)) 
        subAdder1 (
            .clk(clk), 
            .reset(reset), 
            .nums(nums[halfnum*(i-1):(halfnum*i)-1]), 
            .sum(sub_sums[i])
        );
    end
    adder_tree_block # (.NUM(NUM-(halfnum*(MAX_ADDS-1))), .DATA_WIDTH(DATA_WIDTH), .DEPTH(depth - 1)) 
    subAdder2 (
        .clk(clk), 
        .reset(reset), 
        .nums(nums[halfnum*(MAX_ADDS-1):NUM-1]), 
        .sum(sub_sums[0])
    );

    // for (genvar i = 0; i < MAX_ADDS; i++) begin: generate_recurs
    //     localparam add_one = (remainder > i) ? 1: 0;
    //     adder_tree_block # (.NUM(halfnum + add_one), .DATA_WIDTH(DATA_WIDTH), .DEPTH(depth - 1)) 
    //     subAdder1 (
    //         .clk(clk), 
    //         .reset(reset), 
    //         .nums(nums[halfnum*(i-1):(halfnum*i)-1]), 
    //         .sum(sub_sums[i])
    //     );
    // end

    always_comb begin
        total = '0;
        foreach(sub_sums[i]) begin
            total += sub_sums[i];
        end
    end
    always_ff @(posedge clk) begin
        if (reset) sum <= '0;
        else sum <= total;
    end
end
endmodule

function automatic int adder_tree_block_latency(int num);
    localparam int max_adds = 3;
    int latency = (num <= 2) ? 1: $floor(($ln((real'(num) - 1.0) / real'(max_adds)) / $ln(real'(max_adds))) + 2);
    return latency;
endfunction