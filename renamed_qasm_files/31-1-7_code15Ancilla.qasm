OPENQASM 2.0;
include "qelib1.inc";

qreg q[46];
creg rec[181];

reset q[0];
reset q[1];
reset q[2];
reset q[3];
reset q[4];
reset q[5];
reset q[6];
reset q[7];
reset q[8];
reset q[9];
reset q[10];
reset q[11];
reset q[12];
reset q[13];
reset q[14];
reset q[15];
reset q[16];
reset q[17];
reset q[18];
reset q[19];
reset q[20];
reset q[21];
reset q[22];
reset q[23];
reset q[24];
reset q[25];
reset q[26];
reset q[27];
reset q[28];
reset q[29];
reset q[30];
reset q[41];
reset q[42];
reset q[31];
reset q[32];
reset q[33];
reset q[43];
reset q[44];
reset q[45];
reset q[34];
reset q[35];
reset q[36];
reset q[37];
reset q[38];
reset q[39];
reset q[40];
barrier q;

reset q[41]; h q[41]; // decomposed RX
reset q[42]; h q[42]; // decomposed RX
reset q[31]; h q[31]; // decomposed RX
reset q[32]; h q[32]; // decomposed RX
reset q[33]; h q[33]; // decomposed RX
reset q[43]; h q[43]; // decomposed RX
reset q[44]; h q[44]; // decomposed RX
reset q[45]; h q[45]; // decomposed RX
reset q[34]; h q[34]; // decomposed RX
reset q[35]; h q[35]; // decomposed RX
reset q[36]; h q[36]; // decomposed RX
reset q[37]; h q[37]; // decomposed RX
reset q[38]; h q[38]; // decomposed RX
reset q[39]; h q[39]; // decomposed RX
reset q[40]; h q[40]; // decomposed RX
barrier q;

cx q[32], q[2];
cx q[33], q[3];
cx q[34], q[7];
cx q[35], q[8];
cx q[37], q[12];
cx q[38], q[13];
cx q[39], q[14];
cx q[40], q[15];
cx q[41], q[21];
cx q[42], q[22];
cx q[44], q[26];
cx q[45], q[27];
barrier q;

cx q[32], q[4];
cx q[33], q[5];
cx q[34], q[10];
cx q[35], q[11];
cx q[37], q[16];
cx q[38], q[17];
cx q[39], q[18];
cx q[40], q[19];
cx q[41], q[24];
cx q[42], q[25];
cx q[44], q[28];
cx q[45], q[29];
barrier q;

cx q[32], q[1];
cx q[33], q[2];
cx q[34], q[5];
cx q[35], q[7];
cx q[37], q[9];
cx q[38], q[12];
cx q[39], q[11];
cx q[40], q[14];
cx q[41], q[17];
cx q[42], q[21];
cx q[43], q[25];
cx q[44], q[23];
cx q[45], q[26];
barrier q;

cx q[31], q[3];
cx q[32], q[6];
cx q[33], q[4];
cx q[34], q[13];
cx q[35], q[10];
cx q[36], q[15];
cx q[37], q[20];
cx q[38], q[16];
cx q[39], q[22];
cx q[40], q[18];
cx q[41], q[27];
cx q[42], q[24];
cx q[44], q[30];
cx q[45], q[28];
barrier q;

cx q[34], q[4];
cx q[39], q[10];
cx q[41], q[16];
cx q[43], q[24];
barrier q;

cx q[31], q[2];
cx q[34], q[12];
cx q[36], q[14];
cx q[39], q[21];
cx q[41], q[26];
barrier q;

cx q[31], q[0];
cx q[34], q[6];
cx q[36], q[8];
cx q[39], q[13];
cx q[41], q[20];
cx q[43], q[27];
barrier q;

cx q[31], q[1];
cx q[34], q[9];
cx q[36], q[11];
cx q[39], q[17];
cx q[41], q[23];
cx q[43], q[29];
barrier q;

h q[31]; measure q[31] -> rec[0]; h q[31]; // decomposed MX
h q[32]; measure q[32] -> rec[1]; h q[32]; // decomposed MX
h q[33]; measure q[33] -> rec[2]; h q[33]; // decomposed MX
h q[34]; measure q[34] -> rec[3]; h q[34]; // decomposed MX
h q[35]; measure q[35] -> rec[4]; h q[35]; // decomposed MX
h q[36]; measure q[36] -> rec[5]; h q[36]; // decomposed MX
h q[37]; measure q[37] -> rec[6]; h q[37]; // decomposed MX
h q[38]; measure q[38] -> rec[7]; h q[38]; // decomposed MX
h q[39]; measure q[39] -> rec[8]; h q[39]; // decomposed MX
h q[40]; measure q[40] -> rec[9]; h q[40]; // decomposed MX
h q[41]; measure q[41] -> rec[10]; h q[41]; // decomposed MX
h q[42]; measure q[42] -> rec[11]; h q[42]; // decomposed MX
h q[43]; measure q[43] -> rec[12]; h q[43]; // decomposed MX
h q[44]; measure q[44] -> rec[13]; h q[44]; // decomposed MX
h q[45]; measure q[45] -> rec[14]; h q[45]; // decomposed MX
barrier q;

reset q[41];
reset q[42];
reset q[31];
reset q[32];
reset q[33];
reset q[43];
reset q[44];
reset q[45];
reset q[34];
reset q[35];
reset q[36];
reset q[37];
reset q[38];
reset q[39];
reset q[40];
barrier q;

cx q[2], q[32];
cx q[3], q[33];
cx q[7], q[34];
cx q[8], q[35];
cx q[12], q[37];
cx q[13], q[38];
cx q[14], q[39];
cx q[15], q[40];
cx q[21], q[41];
cx q[22], q[42];
cx q[26], q[44];
cx q[27], q[45];
barrier q;

cx q[4], q[32];
cx q[5], q[33];
cx q[10], q[34];
cx q[11], q[35];
cx q[16], q[37];
cx q[17], q[38];
cx q[18], q[39];
cx q[19], q[40];
cx q[24], q[41];
cx q[25], q[42];
cx q[28], q[44];
cx q[29], q[45];
barrier q;

cx q[1], q[32];
cx q[2], q[33];
cx q[5], q[34];
cx q[7], q[35];
cx q[9], q[37];
cx q[12], q[38];
cx q[11], q[39];
cx q[14], q[40];
cx q[17], q[41];
cx q[21], q[42];
cx q[25], q[43];
cx q[23], q[44];
cx q[26], q[45];
barrier q;

cx q[3], q[31];
cx q[6], q[32];
cx q[4], q[33];
cx q[13], q[34];
cx q[10], q[35];
cx q[15], q[36];
cx q[20], q[37];
cx q[16], q[38];
cx q[22], q[39];
cx q[18], q[40];
cx q[27], q[41];
cx q[24], q[42];
cx q[30], q[44];
cx q[28], q[45];
barrier q;

cx q[4], q[34];
cx q[10], q[39];
cx q[16], q[41];
cx q[24], q[43];
barrier q;

cx q[2], q[31];
cx q[12], q[34];
cx q[14], q[36];
cx q[21], q[39];
cx q[26], q[41];
barrier q;

cx q[0], q[31];
cx q[6], q[34];
cx q[8], q[36];
cx q[13], q[39];
cx q[20], q[41];
cx q[27], q[43];
barrier q;

cx q[1], q[31];
cx q[9], q[34];
cx q[11], q[36];
cx q[17], q[39];
cx q[23], q[41];
cx q[29], q[43];
barrier q;

measure q[31] -> rec[15];
measure q[32] -> rec[16];
measure q[33] -> rec[17];
measure q[34] -> rec[18];
measure q[35] -> rec[19];
measure q[36] -> rec[20];
measure q[37] -> rec[21];
measure q[38] -> rec[22];
measure q[39] -> rec[23];
measure q[40] -> rec[24];
measure q[41] -> rec[25];
measure q[42] -> rec[26];
measure q[43] -> rec[27];
measure q[44] -> rec[28];
measure q[45] -> rec[29];
barrier q;

barrier q;

reset q[41]; h q[41]; // decomposed RX
reset q[42]; h q[42]; // decomposed RX
reset q[31]; h q[31]; // decomposed RX
reset q[32]; h q[32]; // decomposed RX
reset q[33]; h q[33]; // decomposed RX
reset q[43]; h q[43]; // decomposed RX
reset q[44]; h q[44]; // decomposed RX
reset q[45]; h q[45]; // decomposed RX
reset q[34]; h q[34]; // decomposed RX
reset q[35]; h q[35]; // decomposed RX
reset q[36]; h q[36]; // decomposed RX
reset q[37]; h q[37]; // decomposed RX
reset q[38]; h q[38]; // decomposed RX
reset q[39]; h q[39]; // decomposed RX
reset q[40]; h q[40]; // decomposed RX
barrier q;

cx q[32], q[2];
cx q[33], q[3];
cx q[34], q[7];
cx q[35], q[8];
cx q[37], q[12];
cx q[38], q[13];
cx q[39], q[14];
cx q[40], q[15];
cx q[41], q[21];
cx q[42], q[22];
cx q[44], q[26];
cx q[45], q[27];
barrier q;

cx q[32], q[4];
cx q[33], q[5];
cx q[34], q[10];
cx q[35], q[11];
cx q[37], q[16];
cx q[38], q[17];
cx q[39], q[18];
cx q[40], q[19];
cx q[41], q[24];
cx q[42], q[25];
cx q[44], q[28];
cx q[45], q[29];
barrier q;

cx q[32], q[1];
cx q[33], q[2];
cx q[34], q[5];
cx q[35], q[7];
cx q[37], q[9];
cx q[38], q[12];
cx q[39], q[11];
cx q[40], q[14];
cx q[41], q[17];
cx q[42], q[21];
cx q[43], q[25];
cx q[44], q[23];
cx q[45], q[26];
barrier q;

cx q[31], q[3];
cx q[32], q[6];
cx q[33], q[4];
cx q[34], q[13];
cx q[35], q[10];
cx q[36], q[15];
cx q[37], q[20];
cx q[38], q[16];
cx q[39], q[22];
cx q[40], q[18];
cx q[41], q[27];
cx q[42], q[24];
cx q[44], q[30];
cx q[45], q[28];
barrier q;

cx q[34], q[4];
cx q[39], q[10];
cx q[41], q[16];
cx q[43], q[24];
barrier q;

cx q[31], q[2];
cx q[34], q[12];
cx q[36], q[14];
cx q[39], q[21];
cx q[41], q[26];
barrier q;

cx q[31], q[0];
cx q[34], q[6];
cx q[36], q[8];
cx q[39], q[13];
cx q[41], q[20];
cx q[43], q[27];
barrier q;

cx q[31], q[1];
cx q[34], q[9];
cx q[36], q[11];
cx q[39], q[17];
cx q[41], q[23];
cx q[43], q[29];
barrier q;

h q[31]; measure q[31] -> rec[30]; h q[31]; // decomposed MX
h q[32]; measure q[32] -> rec[31]; h q[32]; // decomposed MX
h q[33]; measure q[33] -> rec[32]; h q[33]; // decomposed MX
h q[34]; measure q[34] -> rec[33]; h q[34]; // decomposed MX
h q[35]; measure q[35] -> rec[34]; h q[35]; // decomposed MX
h q[36]; measure q[36] -> rec[35]; h q[36]; // decomposed MX
h q[37]; measure q[37] -> rec[36]; h q[37]; // decomposed MX
h q[38]; measure q[38] -> rec[37]; h q[38]; // decomposed MX
h q[39]; measure q[39] -> rec[38]; h q[39]; // decomposed MX
h q[40]; measure q[40] -> rec[39]; h q[40]; // decomposed MX
h q[41]; measure q[41] -> rec[40]; h q[41]; // decomposed MX
h q[42]; measure q[42] -> rec[41]; h q[42]; // decomposed MX
h q[43]; measure q[43] -> rec[42]; h q[43]; // decomposed MX
h q[44]; measure q[44] -> rec[43]; h q[44]; // decomposed MX
h q[45]; measure q[45] -> rec[44]; h q[45]; // decomposed MX
barrier q;

barrier q;

reset q[41];
reset q[42];
reset q[31];
reset q[32];
reset q[33];
reset q[43];
reset q[44];
reset q[45];
reset q[34];
reset q[35];
reset q[36];
reset q[37];
reset q[38];
reset q[39];
reset q[40];
barrier q;

cx q[2], q[32];
cx q[3], q[33];
cx q[7], q[34];
cx q[8], q[35];
cx q[12], q[37];
cx q[13], q[38];
cx q[14], q[39];
cx q[15], q[40];
cx q[21], q[41];
cx q[22], q[42];
cx q[26], q[44];
cx q[27], q[45];
barrier q;

cx q[4], q[32];
cx q[5], q[33];
cx q[10], q[34];
cx q[11], q[35];
cx q[16], q[37];
cx q[17], q[38];
cx q[18], q[39];
cx q[19], q[40];
cx q[24], q[41];
cx q[25], q[42];
cx q[28], q[44];
cx q[29], q[45];
barrier q;

cx q[1], q[32];
cx q[2], q[33];
cx q[5], q[34];
cx q[7], q[35];
cx q[9], q[37];
cx q[12], q[38];
cx q[11], q[39];
cx q[14], q[40];
cx q[17], q[41];
cx q[21], q[42];
cx q[25], q[43];
cx q[23], q[44];
cx q[26], q[45];
barrier q;

cx q[3], q[31];
cx q[6], q[32];
cx q[4], q[33];
cx q[13], q[34];
cx q[10], q[35];
cx q[15], q[36];
cx q[20], q[37];
cx q[16], q[38];
cx q[22], q[39];
cx q[18], q[40];
cx q[27], q[41];
cx q[24], q[42];
cx q[30], q[44];
cx q[28], q[45];
barrier q;

cx q[4], q[34];
cx q[10], q[39];
cx q[16], q[41];
cx q[24], q[43];
barrier q;

cx q[2], q[31];
cx q[12], q[34];
cx q[14], q[36];
cx q[21], q[39];
cx q[26], q[41];
barrier q;

cx q[0], q[31];
cx q[6], q[34];
cx q[8], q[36];
cx q[13], q[39];
cx q[20], q[41];
cx q[27], q[43];
barrier q;

cx q[1], q[31];
cx q[9], q[34];
cx q[11], q[36];
cx q[17], q[39];
cx q[23], q[41];
cx q[29], q[43];
barrier q;

measure q[31] -> rec[45];
measure q[32] -> rec[46];
measure q[33] -> rec[47];
measure q[34] -> rec[48];
measure q[35] -> rec[49];
measure q[36] -> rec[50];
measure q[37] -> rec[51];
measure q[38] -> rec[52];
measure q[39] -> rec[53];
measure q[40] -> rec[54];
measure q[41] -> rec[55];
measure q[42] -> rec[56];
measure q[43] -> rec[57];
measure q[44] -> rec[58];
measure q[45] -> rec[59];
barrier q;

barrier q;

reset q[41]; h q[41]; // decomposed RX
reset q[42]; h q[42]; // decomposed RX
reset q[31]; h q[31]; // decomposed RX
reset q[32]; h q[32]; // decomposed RX
reset q[33]; h q[33]; // decomposed RX
reset q[43]; h q[43]; // decomposed RX
reset q[44]; h q[44]; // decomposed RX
reset q[45]; h q[45]; // decomposed RX
reset q[34]; h q[34]; // decomposed RX
reset q[35]; h q[35]; // decomposed RX
reset q[36]; h q[36]; // decomposed RX
reset q[37]; h q[37]; // decomposed RX
reset q[38]; h q[38]; // decomposed RX
reset q[39]; h q[39]; // decomposed RX
reset q[40]; h q[40]; // decomposed RX
barrier q;

cx q[32], q[2];
cx q[33], q[3];
cx q[34], q[7];
cx q[35], q[8];
cx q[37], q[12];
cx q[38], q[13];
cx q[39], q[14];
cx q[40], q[15];
cx q[41], q[21];
cx q[42], q[22];
cx q[44], q[26];
cx q[45], q[27];
barrier q;

cx q[32], q[4];
cx q[33], q[5];
cx q[34], q[10];
cx q[35], q[11];
cx q[37], q[16];
cx q[38], q[17];
cx q[39], q[18];
cx q[40], q[19];
cx q[41], q[24];
cx q[42], q[25];
cx q[44], q[28];
cx q[45], q[29];
barrier q;

cx q[32], q[1];
cx q[33], q[2];
cx q[34], q[5];
cx q[35], q[7];
cx q[37], q[9];
cx q[38], q[12];
cx q[39], q[11];
cx q[40], q[14];
cx q[41], q[17];
cx q[42], q[21];
cx q[43], q[25];
cx q[44], q[23];
cx q[45], q[26];
barrier q;

cx q[31], q[3];
cx q[32], q[6];
cx q[33], q[4];
cx q[34], q[13];
cx q[35], q[10];
cx q[36], q[15];
cx q[37], q[20];
cx q[38], q[16];
cx q[39], q[22];
cx q[40], q[18];
cx q[41], q[27];
cx q[42], q[24];
cx q[44], q[30];
cx q[45], q[28];
barrier q;

cx q[34], q[4];
cx q[39], q[10];
cx q[41], q[16];
cx q[43], q[24];
barrier q;

cx q[31], q[2];
cx q[34], q[12];
cx q[36], q[14];
cx q[39], q[21];
cx q[41], q[26];
barrier q;

cx q[31], q[0];
cx q[34], q[6];
cx q[36], q[8];
cx q[39], q[13];
cx q[41], q[20];
cx q[43], q[27];
barrier q;

cx q[31], q[1];
cx q[34], q[9];
cx q[36], q[11];
cx q[39], q[17];
cx q[41], q[23];
cx q[43], q[29];
barrier q;

h q[31]; measure q[31] -> rec[60]; h q[31]; // decomposed MX
h q[32]; measure q[32] -> rec[61]; h q[32]; // decomposed MX
h q[33]; measure q[33] -> rec[62]; h q[33]; // decomposed MX
h q[34]; measure q[34] -> rec[63]; h q[34]; // decomposed MX
h q[35]; measure q[35] -> rec[64]; h q[35]; // decomposed MX
h q[36]; measure q[36] -> rec[65]; h q[36]; // decomposed MX
h q[37]; measure q[37] -> rec[66]; h q[37]; // decomposed MX
h q[38]; measure q[38] -> rec[67]; h q[38]; // decomposed MX
h q[39]; measure q[39] -> rec[68]; h q[39]; // decomposed MX
h q[40]; measure q[40] -> rec[69]; h q[40]; // decomposed MX
h q[41]; measure q[41] -> rec[70]; h q[41]; // decomposed MX
h q[42]; measure q[42] -> rec[71]; h q[42]; // decomposed MX
h q[43]; measure q[43] -> rec[72]; h q[43]; // decomposed MX
h q[44]; measure q[44] -> rec[73]; h q[44]; // decomposed MX
h q[45]; measure q[45] -> rec[74]; h q[45]; // decomposed MX
barrier q;

barrier q;

reset q[41];
reset q[42];
reset q[31];
reset q[32];
reset q[33];
reset q[43];
reset q[44];
reset q[45];
reset q[34];
reset q[35];
reset q[36];
reset q[37];
reset q[38];
reset q[39];
reset q[40];
barrier q;

cx q[2], q[32];
cx q[3], q[33];
cx q[7], q[34];
cx q[8], q[35];
cx q[12], q[37];
cx q[13], q[38];
cx q[14], q[39];
cx q[15], q[40];
cx q[21], q[41];
cx q[22], q[42];
cx q[26], q[44];
cx q[27], q[45];
barrier q;

cx q[4], q[32];
cx q[5], q[33];
cx q[10], q[34];
cx q[11], q[35];
cx q[16], q[37];
cx q[17], q[38];
cx q[18], q[39];
cx q[19], q[40];
cx q[24], q[41];
cx q[25], q[42];
cx q[28], q[44];
cx q[29], q[45];
barrier q;

cx q[1], q[32];
cx q[2], q[33];
cx q[5], q[34];
cx q[7], q[35];
cx q[9], q[37];
cx q[12], q[38];
cx q[11], q[39];
cx q[14], q[40];
cx q[17], q[41];
cx q[21], q[42];
cx q[25], q[43];
cx q[23], q[44];
cx q[26], q[45];
barrier q;

cx q[3], q[31];
cx q[6], q[32];
cx q[4], q[33];
cx q[13], q[34];
cx q[10], q[35];
cx q[15], q[36];
cx q[20], q[37];
cx q[16], q[38];
cx q[22], q[39];
cx q[18], q[40];
cx q[27], q[41];
cx q[24], q[42];
cx q[30], q[44];
cx q[28], q[45];
barrier q;

cx q[4], q[34];
cx q[10], q[39];
cx q[16], q[41];
cx q[24], q[43];
barrier q;

cx q[2], q[31];
cx q[12], q[34];
cx q[14], q[36];
cx q[21], q[39];
cx q[26], q[41];
barrier q;

cx q[0], q[31];
cx q[6], q[34];
cx q[8], q[36];
cx q[13], q[39];
cx q[20], q[41];
cx q[27], q[43];
barrier q;

cx q[1], q[31];
cx q[9], q[34];
cx q[11], q[36];
cx q[17], q[39];
cx q[23], q[41];
cx q[29], q[43];
barrier q;

measure q[31] -> rec[75];
measure q[32] -> rec[76];
measure q[33] -> rec[77];
measure q[34] -> rec[78];
measure q[35] -> rec[79];
measure q[36] -> rec[80];
measure q[37] -> rec[81];
measure q[38] -> rec[82];
measure q[39] -> rec[83];
measure q[40] -> rec[84];
measure q[41] -> rec[85];
measure q[42] -> rec[86];
measure q[43] -> rec[87];
measure q[44] -> rec[88];
measure q[45] -> rec[89];
barrier q;

barrier q;

reset q[41]; h q[41]; // decomposed RX
reset q[42]; h q[42]; // decomposed RX
reset q[31]; h q[31]; // decomposed RX
reset q[32]; h q[32]; // decomposed RX
reset q[33]; h q[33]; // decomposed RX
reset q[43]; h q[43]; // decomposed RX
reset q[44]; h q[44]; // decomposed RX
reset q[45]; h q[45]; // decomposed RX
reset q[34]; h q[34]; // decomposed RX
reset q[35]; h q[35]; // decomposed RX
reset q[36]; h q[36]; // decomposed RX
reset q[37]; h q[37]; // decomposed RX
reset q[38]; h q[38]; // decomposed RX
reset q[39]; h q[39]; // decomposed RX
reset q[40]; h q[40]; // decomposed RX
barrier q;

cx q[32], q[2];
cx q[33], q[3];
cx q[34], q[7];
cx q[35], q[8];
cx q[37], q[12];
cx q[38], q[13];
cx q[39], q[14];
cx q[40], q[15];
cx q[41], q[21];
cx q[42], q[22];
cx q[44], q[26];
cx q[45], q[27];
barrier q;

cx q[32], q[4];
cx q[33], q[5];
cx q[34], q[10];
cx q[35], q[11];
cx q[37], q[16];
cx q[38], q[17];
cx q[39], q[18];
cx q[40], q[19];
cx q[41], q[24];
cx q[42], q[25];
cx q[44], q[28];
cx q[45], q[29];
barrier q;

cx q[32], q[1];
cx q[33], q[2];
cx q[34], q[5];
cx q[35], q[7];
cx q[37], q[9];
cx q[38], q[12];
cx q[39], q[11];
cx q[40], q[14];
cx q[41], q[17];
cx q[42], q[21];
cx q[43], q[25];
cx q[44], q[23];
cx q[45], q[26];
barrier q;

cx q[31], q[3];
cx q[32], q[6];
cx q[33], q[4];
cx q[34], q[13];
cx q[35], q[10];
cx q[36], q[15];
cx q[37], q[20];
cx q[38], q[16];
cx q[39], q[22];
cx q[40], q[18];
cx q[41], q[27];
cx q[42], q[24];
cx q[44], q[30];
cx q[45], q[28];
barrier q;

cx q[34], q[4];
cx q[39], q[10];
cx q[41], q[16];
cx q[43], q[24];
barrier q;

cx q[31], q[2];
cx q[34], q[12];
cx q[36], q[14];
cx q[39], q[21];
cx q[41], q[26];
barrier q;

cx q[31], q[0];
cx q[34], q[6];
cx q[36], q[8];
cx q[39], q[13];
cx q[41], q[20];
cx q[43], q[27];
barrier q;

cx q[31], q[1];
cx q[34], q[9];
cx q[36], q[11];
cx q[39], q[17];
cx q[41], q[23];
cx q[43], q[29];
barrier q;

h q[31]; measure q[31] -> rec[90]; h q[31]; // decomposed MX
h q[32]; measure q[32] -> rec[91]; h q[32]; // decomposed MX
h q[33]; measure q[33] -> rec[92]; h q[33]; // decomposed MX
h q[34]; measure q[34] -> rec[93]; h q[34]; // decomposed MX
h q[35]; measure q[35] -> rec[94]; h q[35]; // decomposed MX
h q[36]; measure q[36] -> rec[95]; h q[36]; // decomposed MX
h q[37]; measure q[37] -> rec[96]; h q[37]; // decomposed MX
h q[38]; measure q[38] -> rec[97]; h q[38]; // decomposed MX
h q[39]; measure q[39] -> rec[98]; h q[39]; // decomposed MX
h q[40]; measure q[40] -> rec[99]; h q[40]; // decomposed MX
h q[41]; measure q[41] -> rec[100]; h q[41]; // decomposed MX
h q[42]; measure q[42] -> rec[101]; h q[42]; // decomposed MX
h q[43]; measure q[43] -> rec[102]; h q[43]; // decomposed MX
h q[44]; measure q[44] -> rec[103]; h q[44]; // decomposed MX
h q[45]; measure q[45] -> rec[104]; h q[45]; // decomposed MX
barrier q;

barrier q;

reset q[41];
reset q[42];
reset q[31];
reset q[32];
reset q[33];
reset q[43];
reset q[44];
reset q[45];
reset q[34];
reset q[35];
reset q[36];
reset q[37];
reset q[38];
reset q[39];
reset q[40];
barrier q;

cx q[2], q[32];
cx q[3], q[33];
cx q[7], q[34];
cx q[8], q[35];
cx q[12], q[37];
cx q[13], q[38];
cx q[14], q[39];
cx q[15], q[40];
cx q[21], q[41];
cx q[22], q[42];
cx q[26], q[44];
cx q[27], q[45];
barrier q;

cx q[4], q[32];
cx q[5], q[33];
cx q[10], q[34];
cx q[11], q[35];
cx q[16], q[37];
cx q[17], q[38];
cx q[18], q[39];
cx q[19], q[40];
cx q[24], q[41];
cx q[25], q[42];
cx q[28], q[44];
cx q[29], q[45];
barrier q;

cx q[1], q[32];
cx q[2], q[33];
cx q[5], q[34];
cx q[7], q[35];
cx q[9], q[37];
cx q[12], q[38];
cx q[11], q[39];
cx q[14], q[40];
cx q[17], q[41];
cx q[21], q[42];
cx q[25], q[43];
cx q[23], q[44];
cx q[26], q[45];
barrier q;

cx q[3], q[31];
cx q[6], q[32];
cx q[4], q[33];
cx q[13], q[34];
cx q[10], q[35];
cx q[15], q[36];
cx q[20], q[37];
cx q[16], q[38];
cx q[22], q[39];
cx q[18], q[40];
cx q[27], q[41];
cx q[24], q[42];
cx q[30], q[44];
cx q[28], q[45];
barrier q;

cx q[4], q[34];
cx q[10], q[39];
cx q[16], q[41];
cx q[24], q[43];
barrier q;

cx q[2], q[31];
cx q[12], q[34];
cx q[14], q[36];
cx q[21], q[39];
cx q[26], q[41];
barrier q;

cx q[0], q[31];
cx q[6], q[34];
cx q[8], q[36];
cx q[13], q[39];
cx q[20], q[41];
cx q[27], q[43];
barrier q;

cx q[1], q[31];
cx q[9], q[34];
cx q[11], q[36];
cx q[17], q[39];
cx q[23], q[41];
cx q[29], q[43];
barrier q;

measure q[31] -> rec[105];
measure q[32] -> rec[106];
measure q[33] -> rec[107];
measure q[34] -> rec[108];
measure q[35] -> rec[109];
measure q[36] -> rec[110];
measure q[37] -> rec[111];
measure q[38] -> rec[112];
measure q[39] -> rec[113];
measure q[40] -> rec[114];
measure q[41] -> rec[115];
measure q[42] -> rec[116];
measure q[43] -> rec[117];
measure q[44] -> rec[118];
measure q[45] -> rec[119];
barrier q;

barrier q;

reset q[41]; h q[41]; // decomposed RX
reset q[42]; h q[42]; // decomposed RX
reset q[31]; h q[31]; // decomposed RX
reset q[32]; h q[32]; // decomposed RX
reset q[33]; h q[33]; // decomposed RX
reset q[43]; h q[43]; // decomposed RX
reset q[44]; h q[44]; // decomposed RX
reset q[45]; h q[45]; // decomposed RX
reset q[34]; h q[34]; // decomposed RX
reset q[35]; h q[35]; // decomposed RX
reset q[36]; h q[36]; // decomposed RX
reset q[37]; h q[37]; // decomposed RX
reset q[38]; h q[38]; // decomposed RX
reset q[39]; h q[39]; // decomposed RX
reset q[40]; h q[40]; // decomposed RX
barrier q;

cx q[32], q[2];
cx q[33], q[3];
cx q[34], q[7];
cx q[35], q[8];
cx q[37], q[12];
cx q[38], q[13];
cx q[39], q[14];
cx q[40], q[15];
cx q[41], q[21];
cx q[42], q[22];
cx q[44], q[26];
cx q[45], q[27];
barrier q;

cx q[32], q[4];
cx q[33], q[5];
cx q[34], q[10];
cx q[35], q[11];
cx q[37], q[16];
cx q[38], q[17];
cx q[39], q[18];
cx q[40], q[19];
cx q[41], q[24];
cx q[42], q[25];
cx q[44], q[28];
cx q[45], q[29];
barrier q;

cx q[32], q[1];
cx q[33], q[2];
cx q[34], q[5];
cx q[35], q[7];
cx q[37], q[9];
cx q[38], q[12];
cx q[39], q[11];
cx q[40], q[14];
cx q[41], q[17];
cx q[42], q[21];
cx q[43], q[25];
cx q[44], q[23];
cx q[45], q[26];
barrier q;

cx q[31], q[3];
cx q[32], q[6];
cx q[33], q[4];
cx q[34], q[13];
cx q[35], q[10];
cx q[36], q[15];
cx q[37], q[20];
cx q[38], q[16];
cx q[39], q[22];
cx q[40], q[18];
cx q[41], q[27];
cx q[42], q[24];
cx q[44], q[30];
cx q[45], q[28];
barrier q;

cx q[34], q[4];
cx q[39], q[10];
cx q[41], q[16];
cx q[43], q[24];
barrier q;

cx q[31], q[2];
cx q[34], q[12];
cx q[36], q[14];
cx q[39], q[21];
cx q[41], q[26];
barrier q;

cx q[31], q[0];
cx q[34], q[6];
cx q[36], q[8];
cx q[39], q[13];
cx q[41], q[20];
cx q[43], q[27];
barrier q;

cx q[31], q[1];
cx q[34], q[9];
cx q[36], q[11];
cx q[39], q[17];
cx q[41], q[23];
cx q[43], q[29];
barrier q;

h q[31]; measure q[31] -> rec[120]; h q[31]; // decomposed MX
h q[32]; measure q[32] -> rec[121]; h q[32]; // decomposed MX
h q[33]; measure q[33] -> rec[122]; h q[33]; // decomposed MX
h q[34]; measure q[34] -> rec[123]; h q[34]; // decomposed MX
h q[35]; measure q[35] -> rec[124]; h q[35]; // decomposed MX
h q[36]; measure q[36] -> rec[125]; h q[36]; // decomposed MX
h q[37]; measure q[37] -> rec[126]; h q[37]; // decomposed MX
h q[38]; measure q[38] -> rec[127]; h q[38]; // decomposed MX
h q[39]; measure q[39] -> rec[128]; h q[39]; // decomposed MX
h q[40]; measure q[40] -> rec[129]; h q[40]; // decomposed MX
h q[41]; measure q[41] -> rec[130]; h q[41]; // decomposed MX
h q[42]; measure q[42] -> rec[131]; h q[42]; // decomposed MX
h q[43]; measure q[43] -> rec[132]; h q[43]; // decomposed MX
h q[44]; measure q[44] -> rec[133]; h q[44]; // decomposed MX
h q[45]; measure q[45] -> rec[134]; h q[45]; // decomposed MX
barrier q;

barrier q;

reset q[41];
reset q[42];
reset q[31];
reset q[32];
reset q[33];
reset q[43];
reset q[44];
reset q[45];
reset q[34];
reset q[35];
reset q[36];
reset q[37];
reset q[38];
reset q[39];
reset q[40];
barrier q;

cx q[2], q[32];
cx q[3], q[33];
cx q[7], q[34];
cx q[8], q[35];
cx q[12], q[37];
cx q[13], q[38];
cx q[14], q[39];
cx q[15], q[40];
cx q[21], q[41];
cx q[22], q[42];
cx q[26], q[44];
cx q[27], q[45];
barrier q;

cx q[4], q[32];
cx q[5], q[33];
cx q[10], q[34];
cx q[11], q[35];
cx q[16], q[37];
cx q[17], q[38];
cx q[18], q[39];
cx q[19], q[40];
cx q[24], q[41];
cx q[25], q[42];
cx q[28], q[44];
cx q[29], q[45];
barrier q;

cx q[1], q[32];
cx q[2], q[33];
cx q[5], q[34];
cx q[7], q[35];
cx q[9], q[37];
cx q[12], q[38];
cx q[11], q[39];
cx q[14], q[40];
cx q[17], q[41];
cx q[21], q[42];
cx q[25], q[43];
cx q[23], q[44];
cx q[26], q[45];
barrier q;

cx q[3], q[31];
cx q[6], q[32];
cx q[4], q[33];
cx q[13], q[34];
cx q[10], q[35];
cx q[15], q[36];
cx q[20], q[37];
cx q[16], q[38];
cx q[22], q[39];
cx q[18], q[40];
cx q[27], q[41];
cx q[24], q[42];
cx q[30], q[44];
cx q[28], q[45];
barrier q;

cx q[4], q[34];
cx q[10], q[39];
cx q[16], q[41];
cx q[24], q[43];
barrier q;

cx q[2], q[31];
cx q[12], q[34];
cx q[14], q[36];
cx q[21], q[39];
cx q[26], q[41];
barrier q;

cx q[0], q[31];
cx q[6], q[34];
cx q[8], q[36];
cx q[13], q[39];
cx q[20], q[41];
cx q[27], q[43];
barrier q;

cx q[1], q[31];
cx q[9], q[34];
cx q[11], q[36];
cx q[17], q[39];
cx q[23], q[41];
cx q[29], q[43];
barrier q;

measure q[31] -> rec[135];
measure q[32] -> rec[136];
measure q[33] -> rec[137];
measure q[34] -> rec[138];
measure q[35] -> rec[139];
measure q[36] -> rec[140];
measure q[37] -> rec[141];
measure q[38] -> rec[142];
measure q[39] -> rec[143];
measure q[40] -> rec[144];
measure q[41] -> rec[145];
measure q[42] -> rec[146];
measure q[43] -> rec[147];
measure q[44] -> rec[148];
measure q[45] -> rec[149];
barrier q;

barrier q;

measure q[0] -> rec[150];
measure q[1] -> rec[151];
measure q[2] -> rec[152];
measure q[3] -> rec[153];
measure q[4] -> rec[154];
measure q[5] -> rec[155];
measure q[6] -> rec[156];
measure q[7] -> rec[157];
measure q[8] -> rec[158];
measure q[9] -> rec[159];
measure q[10] -> rec[160];
measure q[11] -> rec[161];
measure q[12] -> rec[162];
measure q[13] -> rec[163];
measure q[14] -> rec[164];
measure q[15] -> rec[165];
measure q[16] -> rec[166];
measure q[17] -> rec[167];
measure q[18] -> rec[168];
measure q[19] -> rec[169];
measure q[20] -> rec[170];
measure q[21] -> rec[171];
measure q[22] -> rec[172];
measure q[23] -> rec[173];
measure q[24] -> rec[174];
measure q[25] -> rec[175];
measure q[26] -> rec[176];
measure q[27] -> rec[177];
measure q[28] -> rec[178];
measure q[29] -> rec[179];
measure q[30] -> rec[180];