using ContextualBandits
using Distributions
using LinearAlgebra
using Random
using Test

@testset "KG functions" begin
    theta = rand(10)
    sigmasq = rand(10)
    sample_std = rand(10)
    Sigma = Diagonal(sigmasq)
    c = cKG(theta,Sigma,sample_std)
    i = iKG(theta,sigmasq,sample_std)
    @test c ≈ i

    sample_std = 0.1
    c = cKG(theta,Sigma,sample_std)
    i = iKG(theta,sigmasq,sample_std)
    @test c ≈ i

    # rng = MersenneTwister(1234)
    # theta = rand(rng,10)
    # Sigma = rand(rng,10,10)
    # Sigma = Sigma * Sigma'
    # sample_std = rand(rng,10)
    theta = [0.5908446386657102, 0.7667970365022592, 0.5662374165061859, 0.4600853424625171, 0.7940257103317943, 0.8541465903790502, 0.20058603493384108, 0.2986142783434118, 0.24683718661000897, 0.5796722333690416]
    Sigma = [3.7557934158806487 2.6058159092720206 2.407118139070727 2.9571617453597714 2.744675170270146 1.1191200590582726 2.1788020704557205 2.571775821360986 2.03861956023372 2.919000378927081;
        2.6058159092720206 3.4169579852787573 2.9476958125707973 2.24533285463492 2.637136443697201 1.080593314566662 1.961199238455767 2.694464482935376 1.9681219717700722 2.696563604608359;
        2.407118139070727 2.9476958125707973 3.0018037243694877 2.0850721956915668 2.5265632485011658 0.9448363431965827 1.7481452587055268 2.512638691120719 1.550579566668791 2.7603468798492266;
        2.9571617453597714 2.24533285463492 2.0850721956915668 3.306578102400961 2.906105105940924 0.8255172770061406 1.4934423143334705 2.5551755985730087 1.9669263274418134 2.410238187612788;
        2.744675170270146 2.637136443697201 2.5265632485011658 2.906105105940924 3.0827925965438885 0.7938478151884486 1.780201809642952 2.6058462931891007 1.7847294618808398 2.761800921070251;
        1.1191200590582726 1.080593314566662 0.9448363431965827 0.8255172770061406 0.7938478151884486 0.782090071968848 0.9186209031704197 1.2048475338132962 1.1944682228636208 1.3510611190522985;
        2.1788020704557205 1.961199238455767 1.7481452587055268 1.4934423143334705 1.780201809642952 0.9186209031704197 2.127590110439943 2.100583879950144 1.4865042332328544 2.383674981975507;
        2.571775821360986 2.694464482935376 2.512638691120719 2.5551755985730087 2.6058462931891007 1.2048475338132962 2.100583879950144 3.1263786398398126 2.0737184567692326 2.976363021886938;
        2.03861956023372 1.9681219717700722 1.550579566668791 1.9669263274418134 1.7847294618808398 1.1944682228636208 1.4865042332328544 2.0737184567692326 2.4803618198718174 1.876500638260746;
        2.919000378927081 2.696563604608359 2.7603468798492266 2.410238187612788 2.761800921070251 1.3510611190522985 2.383674981975507 2.976363021886938 1.876500638260746 4.021474638376733]
    sample_std = [0.4680789803043601, 0.09518151761997728, 0.7277627510365177, 0.982501974805928, 0.4270338006578229, 0.467938838034194, 0.848927018097327, 0.5586466225759212, 0.8074302732679368, 0.013372209281458325]
    c = cKG(theta,Sigma,sample_std)
    @test c == [-0.8846588839665859, -0.7741491951129047, -0.9596368128224648, -0.9792457475294534, -0.7426307677119657, -2.124034868230084, -1.4332461291574268, -1.2420862648959767, -1.977175841884483, -0.8928851843709824]
end