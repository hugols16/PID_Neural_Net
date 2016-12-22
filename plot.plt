set terminal svg enhanced size 1400 900 fname "Times" fsize 36
set output "plot.svg"
set title "PID Neural Net Controller"
set xlabel "t"
set ylabel "y"
plot "./data.dat" using 1:2 title "" with lines lt rgb "blue", "" using 1:3 title "" with lines lt rgb "red"