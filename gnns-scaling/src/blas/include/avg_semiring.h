struct AvgTuple 
{
    float value;
    float weight;
};

void mul(void *z, const void *x, const void *y)
{
    ((AvgTuple *) z)->value = *(float*) x * *(float*) y;
    ((AvgTuple *) z)->weight = 1.;
}

void add(void *z, const void *a, const void *b)
{
    ((AvgTuple*) z)->value *= ((AvgTuple*) a)->weight;
    ((AvgTuple*) z)->weight += ((AvgTuple*) b)->weight;

    ((AvgTuple*) z)->value += ((AvgTuple*) b)->value * ((AvgTuple*) b)->weight;
    ((AvgTuple*) z)->value /= ((AvgTuple*) a)->value;
}

